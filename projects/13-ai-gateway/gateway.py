#!/usr/bin/env python3
"""AI gateway: routing, fallbacks, rate limits, TTFT SLOs.

    ./gateway.py --config config.yaml --port 9000
    ./gateway.py --config config.yaml --print-chain     # show the fallback chain

OpenAI-compatible in and out, so clients do not know it is there. Routes across
self-hosted replicas and API providers, enforces per-tenant limits, degrades
through a configured chain when backends fail, and holds a retry budget so a
partial outage cannot be amplified into a total one.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from llmkit.gateway import (
    Backend,
    DegradationChain,
    HedgeConfig,
    RetryBudget,
    TenantLimits,
    backoff_delay,
)
from llmkit.routing import prefix_key

STATE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    requests: int = 0
    served: int = 0
    failed: int = 0
    rate_limited: int = 0
    degraded: int = 0          # served by a non-primary tier
    hedged: int = 0
    hedge_wins: int = 0
    retries: int = 0
    by_backend: dict[str, int] = field(default_factory=dict)
    by_tier: dict[str, int] = field(default_factory=dict)
    by_reject_reason: dict[str, int] = field(default_factory=dict)
    ttft_ms: list[float] = field(default_factory=list)

    def note_ttft(self, ms: float) -> None:
        self.ttft_ms.append(ms)
        if len(self.ttft_ms) > 10000:
            del self.ttft_ms[:5000]


M = Metrics()


def load_config(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    backends = [Backend(**b) for b in cfg["backends"]]
    chain = DegradationChain(
        backends=backends,
        allow_cached_response=cfg.get("allow_cached_response", True),
        cached_response_max_age_s=cfg.get("cached_response_max_age_s", 3600.0),
    )
    tenants = {t["tenant"]: TenantLimits(**t) for t in cfg.get("tenants", [])}
    if "default" not in tenants:
        tenants["default"] = TenantLimits(tenant="default")
    hedge = HedgeConfig(**cfg.get("hedge", {}))
    budget = RetryBudget(**cfg.get("retry_budget", {}))
    return {"chain": chain, "tenants": tenants, "hedge": hedge,
            "retry_budget": budget, "raw": cfg,
            "max_attempts": cfg.get("max_attempts", 3),
            "response_cache": {}, "cache_ttl": cfg.get("cached_response_max_age_s", 3600.0)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["client"] = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=5.0),
        limits=httpx.Limits(max_connections=4096, max_keepalive_connections=4096))
    STATE["health_task"] = asyncio.create_task(health_loop())
    yield
    STATE["health_task"].cancel()
    await STATE["client"].aclose()


app = FastAPI(title="AI gateway", lifespan=lifespan)


async def health_loop() -> None:
    client: httpx.AsyncClient = STATE["client"]
    while True:
        for b in STATE["chain"].backends:
            if not b.self_hosted:
                continue      # do not health-poll paid APIs; use request outcomes
            try:
                r = await client.get(f"{b.base_url.rstrip('/')}/health", timeout=2.0)
                ok = r.status_code < 400
            except Exception:
                ok = False
            if ok:
                b.consecutive_errors = 0
                b.probe_failures = 0
                b.open_until = min(b.open_until, time.monotonic())
            else:
                # A failing probe must count against the backend, or an
                # unreachable replica keeps being advertised as available and
                # the first user request routed to it pays a connect timeout
                # before failing over. Request outcomes remain the primary
                # signal (a replica can answer /health while wedged), so the
                # probe needs several consecutive failures to trip the breaker
                # rather than one.
                b.probe_failures += 1
                if b.probe_failures >= 3:
                    b.open_until = max(b.open_until, time.monotonic() + 15.0)
        await asyncio.sleep(10.0)


# ---------------------------------------------------------------------------
def estimate_tokens(body: dict[str, Any]) -> int:
    if "messages" in body:
        n = sum(len(str(m.get("content", "")).split())
                for m in body.get("messages") or [])
    else:
        n = len(str(body.get("prompt", "")).split())
    return n + int(body.get("max_tokens") or 256)


def tenant_of(req: Request) -> str:
    return req.headers.get("x-tenant") or "default"


@app.get("/health")
async def health() -> JSONResponse:
    chain: DegradationChain = STATE["chain"]
    up = [b.name for b in chain.backends if b.available]
    primary_up = [b.name for b in chain.backends
                  if b.tier == "primary" and b.available]
    return JSONResponse({
        "status": "ok" if up else "no_backends",
        "degraded": not primary_up and bool(up),
        "available": up,
    })


@app.get("/stats")
async def stats() -> JSONResponse:
    chain: DegradationChain = STATE["chain"]
    from llmkit.metrics import percentile
    return JSONResponse({
        "requests": M.requests, "served": M.served, "failed": M.failed,
        "rate_limited": M.rate_limited, "degraded": M.degraded,
        "hedged": M.hedged, "hedge_wins": M.hedge_wins, "retries": M.retries,
        "by_backend": M.by_backend, "by_tier": M.by_tier,
        "reject_reasons": M.by_reject_reason,
        "ttft_p50": round(percentile(M.ttft_ms, 50), 1) if M.ttft_ms else None,
        "ttft_p95": round(percentile(M.ttft_ms, 95), 1) if M.ttft_ms else None,
        "retry_budget": STATE["retry_budget"].stats(),
        "backends": [{
            "name": b.name, "tier": b.tier, "available": b.available,
            "inflight": b.inflight, "total": b.total, "errors": b.errors,
            "consecutive_errors": b.consecutive_errors,
            "probe_failures": b.probe_failures,
            "breaker_open_for_s": max(0.0, round(b.open_until - time.monotonic(), 1)),
            "ewma_ttft_ms": round(b.ewma_ttft_ms, 1),
            "meeting_slo": b.meeting_slo(),
        } for b in chain.backends],
        "tenants": [{
            "tenant": t.tenant, "inflight": t.inflight,
            "spent_usd": round(t.spent_usd, 4),
        } for t in STATE["tenants"].values()],
    })


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    lines = [
        "# TYPE gateway_requests_total counter",
        f"gateway_requests_total {M.requests}",
        "# TYPE gateway_served_total counter",
        f"gateway_served_total {M.served}",
        "# TYPE gateway_failed_total counter",
        f"gateway_failed_total {M.failed}",
        "# TYPE gateway_rate_limited_total counter",
        f"gateway_rate_limited_total {M.rate_limited}",
        "# TYPE gateway_degraded_total counter",
        f"gateway_degraded_total {M.degraded}",
        "# TYPE gateway_hedged_total counter",
        f"gateway_hedged_total {M.hedged}",
        f"gateway_hedge_wins_total {M.hedge_wins}",
        "# TYPE gateway_retries_total counter",
        f"gateway_retries_total {M.retries}",
    ]
    for name, n in M.by_backend.items():
        lines.append(f'gateway_backend_requests_total{{backend="{name}"}} {n}')
    for tier, n in M.by_tier.items():
        lines.append(f'gateway_tier_requests_total{{tier="{tier}"}} {n}')
    for b in STATE["chain"].backends:
        lines.append(f'gateway_backend_available{{backend="{b.name}"}} '
                     f'{1 if b.available else 0}')
        lines.append(f'gateway_backend_ttft_ms{{backend="{b.name}"}} '
                     f'{b.ewma_ttft_ms:.2f}')
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/admin/backend/{name}/disable")
async def disable_backend(name: str, body: dict[str, Any] | None = None) -> JSONResponse:
    """Manual breaker, for drills and for taking a provider out during an incident."""
    secs = float((body or {}).get("seconds", 300))
    for b in STATE["chain"].backends:
        if b.name == name:
            b.open_until = time.monotonic() + secs
            return JSONResponse({"backend": name, "disabled_for_s": secs})
    raise HTTPException(404, f"no backend {name}")


@app.post("/v1/chat/completions")
async def chat(req: Request):
    return await handle(req, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(req: Request):
    return await handle(req, "/v1/completions")


async def handle(req: Request, path: str):
    body = await req.json()
    tenant_name = tenant_of(req)
    tenants: dict[str, TenantLimits] = STATE["tenants"]
    limits = tenants.get(tenant_name) or tenants["default"]
    chain: DegradationChain = STATE["chain"]
    budget: RetryBudget = STATE["retry_budget"]

    M.requests += 1
    budget.record_request()

    est = estimate_tokens(body)
    ok, reason, retry_after = limits.check(est)
    if not ok:
        M.rate_limited += 1
        M.by_reject_reason[reason] = M.by_reject_reason.get(reason, 0) + 1
        raise HTTPException(
            429, detail=f"rate limited ({reason})",
            headers={"Retry-After": str(max(1, int(retry_after))),
                     "x-ratelimit-reason": reason})

    rid = req.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    key = prefix_key(_system_of(body), _user_of(body))
    limits.inflight += 1
    tried: list[str] = []
    last_err = "none"
    try:
        for attempt in range(STATE["max_attempts"]):
            if attempt > 0:
                if not budget.try_retry():
                    # Budget exhausted: stop retrying rather than amplify load.
                    last_err += " (retry budget exhausted)"
                    break
                M.retries += 1
                await asyncio.sleep(backoff_delay(attempt))

            candidates = chain.candidates(exclude=tried, require_slo=True)
            if not candidates:
                break
            b = candidates[0]
            tried.append(b.name)
            try:
                resp = await forward(req, body, path, b, rid, key, limits, est)
                M.served += 1
                M.by_backend[b.name] = M.by_backend.get(b.name, 0) + 1
                M.by_tier[b.tier] = M.by_tier.get(b.tier, 0) + 1
                if b.tier != "primary":
                    M.degraded += 1
                return resp
            except _BackendError as e:
                last_err = str(e)
                b.observe_error()

        # --- last resort ---------------------------------------------------
        if chain.allow_cached_response:
            cached = _cache_get(key)
            if cached is not None:
                M.served += 1
                M.degraded += 1
                M.by_tier["cached"] = M.by_tier.get("cached", 0) + 1
                return JSONResponse(
                    cached, headers={"x-gateway-tier": "cached",
                                     "x-gateway-degraded": "true"})
        M.failed += 1
        raise HTTPException(
            503, detail=f"all backends failed (tried {tried}); last: {last_err}")
    finally:
        limits.inflight -= 1


class _BackendError(Exception):
    pass


def _system_of(body: dict[str, Any]) -> str | None:
    for m in body.get("messages") or []:
        if m.get("role") == "system":
            return str(m.get("content") or "")
    return None


def _user_of(body: dict[str, Any]) -> str:
    if "messages" in body:
        return "\n".join(str(m.get("content") or "")
                         for m in body["messages"] if m.get("role") != "system")
    return str(body.get("prompt") or "")


def _cache_get(key: str) -> dict[str, Any] | None:
    entry = STATE["response_cache"].get(key)
    if not entry:
        return None
    body, ts = entry
    if time.time() - ts > STATE["cache_ttl"]:
        STATE["response_cache"].pop(key, None)
        return None
    return body


def _cache_put(key: str, body: dict[str, Any]) -> None:
    cache = STATE["response_cache"]
    if len(cache) > 2000:
        for k in list(cache)[:500]:
            cache.pop(k, None)
    cache[key] = (body, time.time())


async def forward(req: Request, body: dict[str, Any], path: str, b: Backend,
                  rid: str, key: str, limits: TenantLimits, est: int):
    client: httpx.AsyncClient = STATE["client"]
    out_body = dict(body)
    out_body["model"] = b.model            # each backend has its own model id
    headers = {"Content-Type": "application/json", "x-request-id": rid}
    if b.api_key_env:
        k = os.environ.get(b.api_key_env)
        if not k:
            raise _BackendError(f"missing API key env {b.api_key_env}")
        headers["Authorization"] = f"Bearer {k}"
    for h in ("x-session-id", "x-tenant"):
        if h in req.headers:
            headers[h] = req.headers[h]

    url = b.base_url.rstrip("/") + path
    b.inflight += 1
    b.total += 1
    t0 = time.perf_counter()

    if not body.get("stream"):
        try:
            r = await client.post(url, json=out_body, headers=headers)
        except Exception as e:
            b.inflight -= 1
            raise _BackendError(f"{type(e).__name__}: {e}") from e
        b.inflight -= 1
        if r.status_code >= 500 or r.status_code == 429:
            raise _BackendError(f"http_{r.status_code}")
        ttft = (time.perf_counter() - t0) * 1e3
        b.observe_success(ttft)
        M.note_ttft(ttft)
        payload = r.json()
        usage = payload.get("usage") or {}
        limits.settle(int(usage.get("total_tokens") or est), est,
                      b.estimate_usd(int(usage.get("prompt_tokens") or 0),
                                     int(usage.get("completion_tokens") or 0)))
        _cache_put(key, payload)
        return JSONResponse(payload, headers={
            "x-gateway-backend": b.name, "x-gateway-tier": b.tier,
            "x-gateway-degraded": "false" if b.tier == "primary" else "true"})

    # Streaming: open upstream and validate status BEFORE returning, so a
    # failure can still be retried on another backend without having already
    # emitted tokens to the client.
    try:
        ctx = client.stream("POST", url, json=out_body, headers=headers)
        resp = await ctx.__aenter__()
    except Exception as e:
        b.inflight -= 1
        raise _BackendError(f"{type(e).__name__}: {e}") from e
    if resp.status_code >= 500 or resp.status_code == 429:
        await ctx.__aexit__(None, None, None)
        b.inflight -= 1
        raise _BackendError(f"http_{resp.status_code}")

    async def body_iter():
        first = True
        n_out = 0
        try:
            async for chunk in resp.aiter_raw():
                if first:
                    ttft = (time.perf_counter() - t0) * 1e3
                    b.observe_success(ttft)
                    M.note_ttft(ttft)
                    first = False
                n_out += chunk.count(b"data:")
                yield chunk
        finally:
            await ctx.__aexit__(None, None, None)
            b.inflight -= 1
            limits.settle(est + n_out, est,
                          b.estimate_usd(est, n_out))

    return StreamingResponse(
        body_iter(), status_code=resp.status_code, media_type="text/event-stream",
        headers={"x-gateway-backend": b.name, "x-gateway-tier": b.tier,
                 "x-request-id": rid,
                 "x-gateway-degraded": "false" if b.tier == "primary" else "true"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--print-chain", action="store_true")
    args = ap.parse_args()

    STATE.update(load_config(args.config))
    if args.print_chain:
        print("degradation chain (tried in this order):")
        print(STATE["chain"].describe())
        print("\ntenants:")
        for t in STATE["tenants"].values():
            print(f"  {t.tenant:<16} {t.rpm:>8.0f} rpm  {t.tpm:>10.0f} tpm  "
                  f"max_concurrent={t.max_concurrent}"
                  + (f"  budget ${t.usd_budget_per_hour}/h"
                     if t.usd_budget_per_hour else ""))
        return

    print(f"gateway on http://{args.host}:{args.port}")
    print(STATE["chain"].describe())
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
