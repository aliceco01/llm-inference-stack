#!/usr/bin/env python3
"""Prefix-aware routing proxy.

    ./proxy.py --backends http://localhost:8001,http://localhost:8002 \
               --router prefix_affinity --port 9000

Routes requests that share a system prompt to the same replica so its KV
blocks get reused. Streams responses through without buffering, so it adds
latency only on the routing decision (microseconds) and not on the token path.

The interesting engineering is in *not* being naive about it. Sending every
request with a popular system prompt to one replica maximises cache hits and
also saturates that replica while the rest idle, at which point the queueing
delay dwarfs the prefill saving. `prefix_affinity` treats the hash as a
preference bounded by load, and deflects when the preferred replica is more
than `--overload-factor` above the fleet mean.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from llmkit.routing import Replica, make_router, prefix_key

STATE: dict[str, Any] = {}


def extract_prefix_material(body: dict[str, Any]) -> tuple[str | None, str]:
    """(system prompt, user text) from either API shape."""
    if "messages" in body:
        sys_txt, parts = None, []
        for m in body.get("messages") or []:
            c = m.get("content") or ""
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if m.get("role") == "system" and sys_txt is None:
                sys_txt = c
            else:
                parts.append(c)
        return sys_txt, "\n".join(parts)
    p = body.get("prompt") or ""
    if isinstance(p, list):
        p = " ".join(map(str, p))
    return None, p


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["client"] = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=5.0),
        limits=httpx.Limits(max_connections=4096, max_keepalive_connections=4096),
    )
    STATE["health_task"] = asyncio.create_task(health_loop())
    yield
    STATE["health_task"].cancel()
    await STATE["client"].aclose()


app = FastAPI(title="prefix-cache proxy", lifespan=lifespan)


async def health_loop() -> None:
    """Passive health checking.

    Active probes alone are not enough: a replica can answer /health while its
    engine is wedged. The router also trips a circuit breaker on consecutive
    request failures, which is the signal that actually correlates with users
    seeing errors.
    """
    client: httpx.AsyncClient = STATE["client"]
    while True:
        for r in STATE["replicas"]:
            try:
                resp = await client.get(f"{r.base_url}/health", timeout=2.0)
                ok = resp.status_code < 400
            except Exception:
                ok = False
            if ok != r.healthy:
                print(f"[health] {r.name} -> {'healthy' if ok else 'UNHEALTHY'}")
            r.healthy = ok
        await asyncio.sleep(STATE["health_interval"])


@app.get("/health")
async def health() -> JSONResponse:
    up = [r.name for r in STATE["replicas"] if r.available]
    return JSONResponse({"status": "ok" if up else "no_backends", "available": up})


@app.get("/v1/models")
async def models() -> JSONResponse:
    client: httpx.AsyncClient = STATE["client"]
    for r in STATE["replicas"]:
        if not r.available:
            continue
        try:
            resp = await client.get(f"{r.base_url}/v1/models", timeout=5.0)
            if resp.status_code < 400:
                return JSONResponse(resp.json())
        except Exception:
            continue
    raise HTTPException(503, "no healthy backend")


@app.get("/stats")
async def stats() -> JSONResponse:
    router = STATE["router"]
    return JSONResponse({
        "router": STATE["router_name"],
        "routing_stats": getattr(router, "stats", {}),
        "replicas": [{
            "name": r.name, "url": r.base_url, "healthy": r.healthy,
            "available": r.available, "inflight": r.inflight,
            "requests": r.total_requests, "errors": r.total_errors,
            "ewma_ttft_ms": round(r.ewma_ttft_ms, 2),
            "known_prefixes": len(r.prefix_keys),
        } for r in STATE["replicas"]],
    })


@app.post("/admin/router")
async def set_router(body: dict[str, Any]) -> JSONResponse:
    """Switch routing strategy without a restart.

    Useful operationally (A/B a strategy against live traffic without a
    rollout) and required by the experiment harness, which must compare
    strategies without running one proxy process per strategy.
    """
    name = body.get("router")
    if not name:
        raise HTTPException(400, "missing 'router'")
    kw: dict[str, Any] = {}
    if name in ("prefix_affinity", "session_affinity"):
        kw["vnodes"] = STATE["vnodes"]
        if name == "prefix_affinity":
            kw["overload_factor"] = body.get("overload_factor", STATE["overload_factor"])
    try:
        STATE["router"] = make_router(name, STATE["replicas"], **kw)
    except KeyError as e:
        raise HTTPException(400, str(e))
    STATE["router_name"] = name
    # Forget believed prefix residency: it describes the old strategy's
    # placement and would bias the new one.
    for r in STATE["replicas"]:
        r.prefix_keys.clear()
    return JSONResponse({"router": name})


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    lines = ["# TYPE proxy_requests_total counter",
             "# TYPE proxy_inflight gauge",
             "# TYPE proxy_errors_total counter"]
    for r in STATE["replicas"]:
        lab = f'{{replica="{r.name}"}}'
        lines += [f"proxy_requests_total{lab} {r.total_requests}",
                  f"proxy_inflight{lab} {r.inflight}",
                  f"proxy_errors_total{lab} {r.total_errors}",
                  f'proxy_replica_ttft_ms{lab} {r.ewma_ttft_ms:.3f}']
    st = getattr(STATE["router"], "stats", {})
    for k, v in st.items():
        lines.append(f"proxy_routing_{k} {v}")
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/chat/completions")
async def chat(req: Request):
    return await handle(req, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(req: Request):
    return await handle(req, "/v1/completions")


async def handle(req: Request, path: str):
    body = await req.json()
    sys_txt, user_txt = extract_prefix_material(body)
    session = req.headers.get("x-session-id")

    # Key choice, in order of how well it predicts cache residency:
    #   session id  -> the whole conversation history is shared
    #   system text -> the shared system prompt is shared
    if STATE["router_name"] == "session_affinity":
        key = session or prefix_key(sys_txt, user_txt)
    else:
        key = prefix_key(sys_txt, user_txt) if (sys_txt or user_txt) else None

    est_tokens = len((sys_txt or "").split()) + len(user_txt.split())
    rid = req.headers.get("x-request-id") or uuid.uuid4().hex[:12]

    last_err = None
    tried: list[str] = []
    for attempt in range(STATE["max_retries"] + 1):
        candidates = [r for r in STATE["replicas"] if r.name not in tried]
        replica = STATE["router"].pick(candidates, key, est_tokens)
        if replica is None:
            raise HTTPException(503, f"no healthy backend (tried {tried}); last: {last_err}")
        tried.append(replica.name)
        try:
            return await forward(req, body, path, replica, key, rid, est_tokens)
        except _BackendError as e:
            last_err = str(e)
            replica.observe_error()
            # Retrying a streamed response that already emitted tokens would
            # duplicate output, so forward() only raises before the first byte.
            if attempt >= STATE["max_retries"]:
                raise HTTPException(502, f"all backends failed: {last_err}")
    raise HTTPException(502, "unreachable")


class _BackendError(Exception):
    pass


async def forward(req: Request, body: dict[str, Any], path: str,
                  replica: Replica, key: str | None, rid: str, est_tokens: int):
    client: httpx.AsyncClient = STATE["client"]
    url = replica.base_url.rstrip("/") + path
    headers = {"Content-Type": "application/json", "x-request-id": rid}
    for h in ("authorization", "x-session-id", "x-tenant"):
        if h in req.headers:
            headers[h] = req.headers[h]

    replica.inflight += 1
    replica.queued_tokens += est_tokens
    replica.total_requests += 1
    t0 = time.perf_counter()

    if not body.get("stream"):
        try:
            resp = await client.post(url, json=body, headers=headers)
        except Exception as e:
            replica.inflight -= 1
            replica.queued_tokens -= est_tokens
            raise _BackendError(f"{type(e).__name__}: {e}") from e
        replica.inflight -= 1
        replica.queued_tokens -= est_tokens
        if resp.status_code >= 500:
            raise _BackendError(f"http_{resp.status_code}")
        replica.observe_success()
        if key:
            replica.note_prefix(key)
        replica.observe_ttft((time.perf_counter() - t0) * 1e3)
        return JSONResponse(json.loads(resp.text), status_code=resp.status_code,
                            headers={"x-replica": replica.name, "x-routing-key": key or ""})

    # Streaming path. The upstream request is opened here so a connection
    # failure or a 5xx status is raised BEFORE any bytes reach the client,
    # which is what makes the retry above safe.
    try:
        ctx = client.stream("POST", url, json=body, headers=headers)
        resp = await ctx.__aenter__()
    except Exception as e:
        replica.inflight -= 1
        replica.queued_tokens -= est_tokens
        raise _BackendError(f"{type(e).__name__}: {e}") from e

    if resp.status_code >= 500:
        await ctx.__aexit__(None, None, None)
        replica.inflight -= 1
        replica.queued_tokens -= est_tokens
        raise _BackendError(f"http_{resp.status_code}")

    async def body_iter():
        first = True
        try:
            async for chunk in resp.aiter_raw():
                if first:
                    replica.observe_ttft((time.perf_counter() - t0) * 1e3)
                    first = False
                yield chunk
        finally:
            await ctx.__aexit__(None, None, None)
            replica.inflight -= 1
            replica.queued_tokens -= est_tokens
            replica.observe_success()
            if key:
                replica.note_prefix(key)

    return StreamingResponse(
        body_iter(), status_code=resp.status_code, media_type="text/event-stream",
        headers={"x-replica": replica.name, "x-routing-key": (key or "")[:16],
                 "x-request-id": rid},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", required=True,
                    help="comma-separated base URLs")
    ap.add_argument("--router", default="prefix_affinity",
                    choices=["prefix_affinity", "session_affinity", "least_loaded",
                             "round_robin", "random"])
    ap.add_argument("--overload-factor", type=float, default=1.25,
                    help="deflect from the hashed replica when its inflight exceeds "
                         "this multiple of the fleet mean")
    ap.add_argument("--vnodes", type=int, default=160)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--health-interval", type=float, default=5.0)
    args = ap.parse_args()

    urls = [u.strip() for u in args.backends.split(",") if u.strip()]
    replicas = [Replica(name=f"r{i}", base_url=u) for i, u in enumerate(urls)]
    kw: dict[str, Any] = {}
    if args.router in ("prefix_affinity", "session_affinity"):
        kw["vnodes"] = args.vnodes
        if args.router == "prefix_affinity":
            kw["overload_factor"] = args.overload_factor
    STATE.update({
        "replicas": replicas,
        "router": make_router(args.router, replicas, **kw),
        "router_name": args.router,
        "max_retries": args.max_retries,
        "health_interval": args.health_interval,
        "vnodes": args.vnodes,
        "overload_factor": args.overload_factor,
    })
    print(f"prefix proxy on http://{args.host}:{args.port}  router={args.router}")
    for r in replicas:
        print(f"  backend {r.name}: {r.base_url}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
