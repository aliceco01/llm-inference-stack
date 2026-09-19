#!/usr/bin/env python3
"""OpenAI-compatible server backed by the engine simulator, in real time.

Why this exists: every other project in this repo needs something to point at.
Requiring an H100 to test a gateway's retry logic or an autoscaler's queue
polling would make 12 of the 15 projects undevelopable without a GPU bill.

This is a genuine async HTTP server. It streams real SSE frames over real
sockets with inter-token spacing produced by the roofline model, and it exposes
the same Prometheus metric names vLLM does, so the benchmark harness, the KV
monitor, the prefix-cache proxy, the autoscaler and the chaos suite all work
against it unmodified and then work against real vLLM unmodified.

What it is not: a model. It emits filler tokens. Use it to develop and test
serving infrastructure, never to make a quality claim.

    python mock_server.py --model llama-3.1-8b --gpu h100-sxm --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from llmkit.simulator import EngineConfig, EngineSim, SimRequest
from llmkit.simulator.engine import PreemptionMode

FILLER = ["the", "model", "produces", "tokens", "here", "and", "the", "exact", "text", "is", "irrelevant", "because", "this", "server", "measures", "scheduling", "and", "latency", "behaviour", "not", "output", "quality"]


@dataclass
class Pending:
    """Server-side view of one in-flight HTTP request."""

    request_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    emitted: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    max_tokens: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class RealTimeEngine:
    """Drives EngineSim on a wall clock and fans tokens out to HTTP streams."""

    def __init__(self, cfg: EngineConfig, *, speedup: float = 1.0,
                 fault: FaultState | None = None) -> None:
        self.sim = EngineSim(cfg)
        self.cfg = cfg
        self.speedup = speedup
        self.fault = fault or FaultState()
        self.pending: dict[str, Pending] = {}
        self.started = time.time()
        self._task: asyncio.Task | None = None
        self._seen: dict[str, int] = {}
        # Prometheus-style counters, named to match vLLM.
        self.m_prompt_tokens = 0
        self.m_generation_tokens = 0
        self.m_finished = 0
        self.m_ttft: list[float] = []
        self.m_e2e: list[float] = []
        self.m_prefix_queries = 0
        self.m_prefix_hits = 0
        self._arrivals: dict[str, float] = {}

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def submit(self, p: Pending, sim_req: SimRequest) -> None:
        self.pending[p.request_id] = p
        self._arrivals[p.request_id] = time.time()
        # Arrival is "now" on the simulator clock: the engine is being driven
        # in real time, so its clock tracks wall time.
        sim_req.arrival_ms = self.sim.now_ms
        self.sim.submit([sim_req])

    async def _loop(self) -> None:
        """Step the engine, then sleep for however long that step took."""
        while True:
            if self.fault.paused:
                await asyncio.sleep(0.02)
                continue
            t0 = time.perf_counter()
            before = self.sim.now_ms
            tr = self.sim.step()
            if tr is None:
                await asyncio.sleep(0.002)   # idle
                continue
            dt_ms = (self.sim.now_ms - before) * self.fault.slowdown

            # Sleep for the step's duration BEFORE releasing its tokens. The
            # engine's virtual clock says this work completes at now_ms, so
            # publishing the tokens first would hand the client its first token
            # a full step early and under-report TTFT by exactly that step.
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            sleep_s = max(0.0, (dt_ms / self.speedup - elapsed_ms) / 1e3)
            if sleep_s:
                await asyncio.sleep(sleep_s)
            self._drain()

    def _drain(self) -> None:
        """Push newly generated tokens into each request's SSE queue."""
        for s in list(self.sim.running) + self.sim.finished[-256:]:
            rid = s.req.request_id
            p = self.pending.get(rid)
            if p is None:
                continue
            seen = self._seen.get(rid, 0)
            if s.generated > seen:
                for _ in range(s.generated - seen):
                    p.queue.put_nowait(FILLER[(p.emitted + seen) % len(FILLER)] + " ")
                self._seen[rid] = s.generated
                p.cached_tokens = s.cached_tokens
            if s.done and not p.done.is_set():
                p.done.set()
                self.m_finished += 1
                self.m_prompt_tokens += s.req.prompt_tokens
                self.m_generation_tokens += s.generated
                self.m_prefix_queries += s.req.prompt_tokens
                self.m_prefix_hits += s.cached_tokens
                t_arr = self._arrivals.pop(rid, None)
                if t_arr:
                    self.m_e2e.append(time.time() - t_arr)
                    if s.first_token_ms > 0:
                        self.m_ttft.append(
                            max((s.first_token_ms - s.arrival_ms) / 1e3, 0.0)
                        )

    # ------------------------------------------------------------------
    def metrics_text(self) -> str:
        """Prometheus exposition using vLLM's metric names.

        Matching the names exactly is the point: the KV monitor (project 03)
        and the autoscaler (project 11) are written against these and must not
        need a code change to point at production.
        """
        sim = self.sim
        lines = [
            "# HELP vllm:num_requests_running Number of requests currently running.",
            "# TYPE vllm:num_requests_running gauge",
            f'vllm:num_requests_running{{model_name="{self.cfg.model}"}} {len(sim.running)}',
            "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
            "# TYPE vllm:num_requests_waiting gauge",
            f'vllm:num_requests_waiting{{model_name="{self.cfg.model}"}} {len(sim.waiting) + len(sim._pending)}',
            "# HELP vllm:num_requests_swapped Number of requests swapped to CPU.",
            "# TYPE vllm:num_requests_swapped gauge",
            f'vllm:num_requests_swapped{{model_name="{self.cfg.model}"}} {len(sim.swapped)}',
            "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent.",
            "# TYPE vllm:gpu_cache_usage_perc gauge",
            f'vllm:gpu_cache_usage_perc{{model_name="{self.cfg.model}"}} {sim.kv.utilization:.6f}',
            "# HELP vllm:num_preemptions_total Cumulative number of preemptions.",
            "# TYPE vllm:num_preemptions_total counter",
            f'vllm:num_preemptions_total{{model_name="{self.cfg.model}"}} {sim.total_preemptions}',
            "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.",
            "# TYPE vllm:prompt_tokens_total counter",
            f'vllm:prompt_tokens_total{{model_name="{self.cfg.model}"}} {self.m_prompt_tokens}',
            "# HELP vllm:generation_tokens_total Number of generation tokens processed.",
            "# TYPE vllm:generation_tokens_total counter",
            f'vllm:generation_tokens_total{{model_name="{self.cfg.model}"}} {self.m_generation_tokens}',
            "# HELP vllm:request_success_total Count of successfully processed requests.",
            "# TYPE vllm:request_success_total counter",
            f'vllm:request_success_total{{model_name="{self.cfg.model}"}} {self.m_finished}',
            "# HELP vllm:prefix_cache_queries_total Prefix cache queries, in tokens.",
            "# TYPE vllm:prefix_cache_queries_total counter",
            f'vllm:prefix_cache_queries_total{{model_name="{self.cfg.model}"}} {self.m_prefix_queries}',
            "# HELP vllm:prefix_cache_hits_total Prefix cache hits, in tokens.",
            "# TYPE vllm:prefix_cache_hits_total counter",
            f'vllm:prefix_cache_hits_total{{model_name="{self.cfg.model}"}} {self.m_prefix_hits}',
            "# HELP sim:num_gpu_blocks Total KV blocks in the pool.",
            "# TYPE sim:num_gpu_blocks gauge",
            f"sim:num_gpu_blocks {sim.kv.num_blocks}",
            "# HELP sim:num_gpu_blocks_used KV blocks currently allocated.",
            "# TYPE sim:num_gpu_blocks_used gauge",
            f"sim:num_gpu_blocks_used {sim.kv.num_used}",
            "# HELP sim:num_gpu_blocks_evictable Cached blocks with refcount 0, reclaimable on demand.",
            "# TYPE sim:num_gpu_blocks_evictable gauge",
            f"sim:num_gpu_blocks_evictable {sim.kv.num_evictable}",
        ]
        for name, vals in (("vllm:time_to_first_token_seconds", self.m_ttft),
                           ("vllm:e2e_request_latency_seconds", self.m_e2e)):
            lines += [f"# HELP {name} Histogram.", f"# TYPE {name} histogram"]
            buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
            for b in buckets:
                lines.append(f'{name}_bucket{{le="{b}"}} {sum(1 for v in vals if v <= b)}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {len(vals)}')
            lines.append(f"{name}_count {len(vals)}")
            lines.append(f"{name}_sum {sum(vals):.6f}")
        return "\n".join(lines) + "\n"


@dataclass
class FaultState:
    """Knobs the chaos suite (project 14) drives over HTTP."""

    slowdown: float = 1.0      # multiply every step duration
    paused: bool = False       # simulate a stalled/evicted replica
    error_rate: float = 0.0    # fraction of requests rejected with 503
    ttft_jitter_ms: float = 0.0


ENGINE: RealTimeEngine | None = None


def estimate_prompt_tokens(body: dict[str, Any]) -> tuple[int, str, str | None]:
    """Token count plus the text used for prefix identity."""
    if "messages" in body:
        parts, sys_txt = [], None
        for m in body["messages"]:
            c = m.get("content") or ""
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if m.get("role") == "system" and sys_txt is None:
                sys_txt = c
            parts.append(c)
        full = "\n".join(parts)
        return max(1, len(full.split())), full, sys_txt
    p = body.get("prompt") or ""
    if isinstance(p, list):
        p = " ".join(map(str, p))
    return max(1, len(p.split())), p, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert ENGINE is not None
    ENGINE.start()
    yield
    await ENGINE.stop()


app = FastAPI(title="llmkit mock inference server", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "simulated": True})


@app.get("/v1/models")
async def models() -> JSONResponse:
    assert ENGINE
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": ENGINE.cfg.model, "object": "model", "owned_by": "llmkit-sim",
            "max_model_len": ENGINE.cfg.max_model_len,
            "simulated": True,
        }],
    })


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    assert ENGINE
    return PlainTextResponse(ENGINE.metrics_text(), media_type="text/plain; version=0.0.4")


@app.get("/sim/stats")
async def sim_stats() -> JSONResponse:
    assert ENGINE
    return JSONResponse(ENGINE.sim.stats())


@app.post("/sim/reset")
async def sim_reset() -> JSONResponse:
    """Drop the KV cache and all counters.

    Required for honest A/B routing experiments: without it, the second
    strategy under test inherits a cache warmed by the first and reports a hit
    rate it did not earn.
    """
    assert ENGINE
    e = ENGINE
    cfg = e.cfg
    e.sim = EngineSim(cfg)
    e.pending.clear()
    e._seen.clear()
    e._arrivals.clear()
    e.m_prompt_tokens = e.m_generation_tokens = e.m_finished = 0
    e.m_prefix_queries = e.m_prefix_hits = 0
    e.m_ttft.clear(); e.m_e2e.clear()
    return JSONResponse({"reset": True, "num_gpu_blocks": e.sim.kv.num_blocks})


@app.post("/sim/fault")
async def set_fault(body: dict[str, Any]) -> JSONResponse:
    """Chaos control plane. Project 14 drives this."""
    assert ENGINE
    f = ENGINE.fault
    for k in ("slowdown", "paused", "error_rate", "ttft_jitter_ms"):
        if k in body:
            setattr(f, k, body[k])
    return JSONResponse({"slowdown": f.slowdown, "paused": f.paused,
                         "error_rate": f.error_rate, "ttft_jitter_ms": f.ttft_jitter_ms})


@app.post("/v1/chat/completions")
async def chat(req: FastAPIRequest):
    return await _handle(req, api="chat")


@app.post("/v1/completions")
async def completions(req: FastAPIRequest):
    return await _handle(req, api="completions")


async def _handle(http_req: FastAPIRequest, api: str):
    assert ENGINE
    import random
    body = await http_req.json()
    if ENGINE.fault.error_rate and random.random() < ENGINE.fault.error_rate:
        raise HTTPException(status_code=503, detail="injected fault: replica unhealthy")

    n_prompt, text, sys_txt = estimate_prompt_tokens(body)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 128)
    if n_prompt + max_tokens > ENGINE.cfg.max_model_len:
        raise HTTPException(
            status_code=400,
            detail=f"prompt ({n_prompt}) + max_tokens ({max_tokens}) exceeds "
                   f"max_model_len ({ENGINE.cfg.max_model_len})",
        )
    rid = http_req.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    session = http_req.headers.get("x-session-id")
    tenant = http_req.headers.get("x-tenant")

    # Prefix identity: the system prompt defines the shared prefix, which is
    # what the routing layer in project 04 keys on.
    shared = len(sys_txt.split()) if sys_txt else 0
    sim_req = SimRequest(
        request_id=rid, prompt_tokens=n_prompt, output_tokens=max_tokens,
        session_id=session, tenant=tenant,
        shared_prefix_tokens=min(shared, n_prompt),
        prefix_group=(f"sys:{hash(sys_txt) & 0xffffff}" if sys_txt else None),
    )
    p = Pending(request_id=rid, prompt_tokens=n_prompt, max_tokens=max_tokens)
    ENGINE.submit(p, sim_req)

    if not body.get("stream"):
        return await _non_streaming(p, api, rid)
    return StreamingResponse(
        _sse(p, api, rid), media_type="text/event-stream",
        headers={"x-replica": os.environ.get("REPLICA_NAME", "sim-0"),
                 "x-request-id": rid},
    )


async def _non_streaming(p: Pending, api: str, rid: str) -> JSONResponse:
    assert ENGINE
    chunks: list[str] = []
    while not p.done.is_set() or not p.queue.empty():
        try:
            chunks.append(await asyncio.wait_for(p.queue.get(), timeout=120))
        except asyncio.TimeoutError:
            break
    ENGINE.pending.pop(rid, None)
    text = "".join(chunks)
    usage = {"prompt_tokens": p.prompt_tokens, "completion_tokens": len(chunks),
             "total_tokens": p.prompt_tokens + len(chunks),
             "prompt_tokens_details": {"cached_tokens": p.cached_tokens}}
    body = {
        "id": f"cmpl-{rid}", "object": "chat.completion", "created": int(time.time()),
        "model": ENGINE.cfg.model, "usage": usage,
        "choices": [{"index": 0, "finish_reason": "length",
                     **({"message": {"role": "assistant", "content": text}}
                        if api == "chat" else {"text": text})}],
    }
    return JSONResponse(body, headers={"x-replica": os.environ.get("REPLICA_NAME", "sim-0")})


async def _sse(p: Pending, api: str, rid: str) -> AsyncIterator[str]:
    assert ENGINE
    created = int(time.time())
    model = ENGINE.cfg.model

    def frame(delta: dict[str, Any] | None, finish: str | None = None,
              usage: dict[str, Any] | None = None) -> str:
        choice: dict[str, Any] = {"index": 0, "finish_reason": finish}
        if api == "chat":
            choice["delta"] = delta or {}
        else:
            choice["text"] = (delta or {}).get("content", "")
        obj = {"id": f"cmpl-{rid}", "object": "chat.completion.chunk",
               "created": created, "model": model, "choices": [choice]}
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n"

    # Role-only first frame, exactly as real OpenAI-compatible servers send.
    # The benchmark client is built to not count this as the first token.
    yield frame({"role": "assistant"})
    n = 0
    try:
        while True:
            if p.done.is_set() and p.queue.empty():
                break
            try:
                tok = await asyncio.wait_for(p.queue.get(), timeout=120)
            except asyncio.TimeoutError:
                break
            n += 1
            yield frame({"content": tok})
    except asyncio.CancelledError:
        p.cancelled = True
        raise
    finally:
        ENGINE.pending.pop(rid, None)
    usage = {"prompt_tokens": p.prompt_tokens, "completion_tokens": n,
             "total_tokens": p.prompt_tokens + n,
             "prompt_tokens_details": {"cached_tokens": p.cached_tokens}}
    yield frame({}, finish="length", usage=usage)
    yield "data: [DONE]\n\n"


def build_engine(args: argparse.Namespace) -> RealTimeEngine:
    cfg = EngineConfig(
        model=args.model, gpu=args.gpu, tp=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=not args.no_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        preemption_mode=PreemptionMode(args.preemption_mode),
        spec_draft_tokens=args.spec_draft_tokens,
        spec_acceptance_rate=args.spec_acceptance_rate,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        kv_dtype=args.kv_dtype, weight_dtype=args.weight_dtype,
    )
    return RealTimeEngine(cfg, speedup=args.speedup)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--gpu", default="h100-sxm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--enable-chunked-prefill", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--preemption-mode", default="recompute", choices=["recompute", "swap"])
    ap.add_argument("--weight-dtype", default="bf16")
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--spec-draft-tokens", type=int, default=0)
    ap.add_argument("--spec-acceptance-rate", type=float, default=0.0)
    ap.add_argument("--num-gpu-blocks-override", type=int, default=None)
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="run the virtual clock faster than real time (1.0 = real time)")
    args = ap.parse_args()

    global ENGINE
    ENGINE = build_engine(args)
    print(ENGINE.sim.budget.explain())
    print(f"\nserving {args.model} on simulated {args.gpu} at "
          f"http://{args.host}:{args.port}  (SIMULATED, speedup={args.speedup}x)")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
