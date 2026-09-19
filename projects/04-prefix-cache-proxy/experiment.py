#!/usr/bin/env python3
"""Measure what prefix-aware routing actually buys.

    ./experiment.py --replicas 4 --routers round_robin,least_loaded,prefix_affinity,session_affinity

Spins up N simulated backends and a proxy per routing strategy, drives an
identical shared-prefix workload through each, and reports TTFT and prefix
cache hit rate side by side. Everything runs locally; no GPU required.

The workload matters more than the router. Prefix routing can only help when
prefixes actually repeat, so the script runs three shapes:

  rag        a few large system prompts shared by many requests
  multiturn  conversations whose history grows and is fully reused
  unique     no shared prefix at all, the control that should show no gain

If a routing change shows a gain on `unique`, the experiment is measuring noise.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llmkit import (
    SLO,
    EndpointConfig,
    StreamingClient,
    WorkloadGenerator,
    report,
    run_closed_loop,
    summarize,
)
from llmkit.workload import LengthSpec, WorkloadSpec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SERVER = REPO / "projects" / "01-inference-server" / "mock_server.py"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Proc:
    def __init__(self, cmd: list[str], name: str) -> None:
        self.name = name
        self.p = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def stop(self) -> None:
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
        except Exception:
            self.p.terminate()


async def wait_ready(url: str, timeout: float = 45.0) -> bool:
    import httpx
    t0 = time.time()
    async with httpx.AsyncClient(timeout=2.0) as c:
        while time.time() - t0 < timeout:
            try:
                if (await c.get(f"{url}/health")).status_code < 400:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return False


WORKLOADS = {
    "rag": WorkloadSpec(
        name="rag", input_len=LengthSpec("fixed", 256), output_len=LengthSpec("fixed", 32),
        system_prompt_tokens=2048, n_prefix_variants=4, multi_turn=1,
    ),
    "multiturn": WorkloadSpec(
        name="multiturn", input_len=LengthSpec("fixed", 200),
        output_len=LengthSpec("fixed", 32), system_prompt_tokens=512,
        n_prefix_variants=2, multi_turn=6,
    ),
    "unique": WorkloadSpec(
        name="unique", input_len=LengthSpec("fixed", 1024),
        output_len=LengthSpec("fixed", 32), system_prompt_tokens=0, multi_turn=1,
    ),
}


async def switch_router(proxy_url: str, router: str) -> None:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(f"{proxy_url}/admin/router", json={"router": router})
        r.raise_for_status()


async def reset_backends(urls: list[str]) -> None:
    """Clear every backend's KV cache and counters between strategies."""
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as c:
        await asyncio.gather(*[c.post(f"{u}/sim/reset") for u in urls],
                             return_exceptions=True)


async def drive(proxy_url: str, spec: WorkloadSpec, *, concurrency: int,
                n_requests: int, warmup: int, label: str) -> Any:
    ep = EndpointConfig(base_url=proxy_url, model="llama-3.1-8b")
    slo = SLO(ttft_ms=1000, p_itl_ms=50)
    async with StreamingClient(ep) as client:
        gen = WorkloadGenerator(spec)
        res = await run_closed_loop(
            client, gen.stream(n_requests + warmup),
            concurrency=concurrency, max_requests=n_requests + warmup,
        )
    s = summarize(res.records, label=label, concurrency=concurrency, slo=slo,
                  warmup_requests=warmup)
    from collections import Counter
    s.meta["replica_spread"] = dict(Counter(r.replica or "?" for r in res.records if r.ok))
    return s


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--routers", default="round_robin,least_loaded,prefix_affinity,session_affinity")
    ap.add_argument("--workloads", default="rag,multiturn,unique")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=192)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=768,
                    help="KV blocks per replica (768 x 16 = 12,288 tokens). Small "
                         "on purpose: routing only matters under cache pressure.")
    ap.add_argument("--out", default="results")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    procs: list[Proc] = []
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # --- backends -----------------------------------------------------
        backend_ports = [free_port() for _ in range(args.replicas)]
        for i, port in enumerate(backend_ports):
            # KV cache is deliberately constrained. With an unconstrained
            # H100 cache every replica holds every prefix, so routing cannot
            # matter and the experiment measures nothing. Prefix routing is
            # only worth building when the working set exceeds per-replica
            # cache capacity, which is exactly the regime set here.
            env_cmd = [args.python, str(SERVER), "--model", "llama-3.1-8b",
                       "--gpu", "h100-sxm", "--max-model-len", "4096",
                       "--num-gpu-blocks-override", str(args.blocks),
                       "--port", str(port), "--host", "127.0.0.1"]
            procs.append(Proc(env_cmd, f"backend{i}"))
        print(f"starting {args.replicas} backends on {backend_ports} ...")
        for port in backend_ports:
            if not await wait_ready(f"http://127.0.0.1:{port}"):
                print(f"backend on {port} failed to start", file=sys.stderr)
                return 2
        backend_urls = [f"http://127.0.0.1:{p}" for p in backend_ports]
        backends = ",".join(backend_urls)

        # --- a single proxy, switched between strategies -------------------
        # One process rather than one per strategy: 4 backends plus 4 proxies
        # plus the driver exceeds available RAM on a small machine, and the
        # extra processes buy nothing since the runs are sequential anyway.
        routers = [r.strip() for r in args.routers.split(",") if r.strip()]
        proxy_port = free_port()
        procs.append(Proc(
            [args.python, str(HERE / "proxy.py"), "--backends", backends,
             "--router", routers[0], "--port", str(proxy_port), "--host", "127.0.0.1"],
            "proxy"))
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        print("starting proxy ...")
        if not await wait_ready(proxy_url):
            print("proxy failed to start", file=sys.stderr)
            return 2

        # --- run ----------------------------------------------------------
        results: dict[str, dict[str, Any]] = {}
        for wname in [w.strip() for w in args.workloads.split(",")]:
            spec = WORKLOADS[wname]
            print(f"\n=== workload: {wname} ===")
            results[wname] = {}
            for r in routers:
                # Isolation: every strategy must start from a cold cache, or the
                # first one to run silently warms the cache for the rest.
                await switch_router(proxy_url, r)
                await reset_backends(backend_urls)
                s = await drive(proxy_url, spec, concurrency=args.concurrency,
                                n_requests=args.requests, warmup=args.warmup,
                                label=f"{r}")
                results[wname][r] = s
                spread = s.meta.get("replica_spread", {})
                bal = (f"{min(spread.values())}-{max(spread.values())}"
                       if spread else "?")
                print(f"  {r:<18} TTFT p50 {s.ttft.p50:7.1f}  p95 {s.ttft.p95:7.1f}ms   "
                      f"hit {s.prefix_hit_rate*100:5.1f}%   "
                      f"out {s.output_tok_per_s:7.1f} tok/s   spread {bal}")

        # --- report -------------------------------------------------------
        md = ["# Prefix-cache routing: measured effect", "",
              f"- replicas: {args.replicas} (simulated llama-3.1-8b on h100-sxm)",
              f"- concurrency: {args.concurrency}, requests per point: {args.requests}",
              "- **simulated**: timings come from the roofline model, not hardware", ""]
        for wname, per_router in results.items():
            md += [f"## Workload: `{wname}`", ""]
            base = per_router.get("round_robin")
            rows = []
            for r, s in per_router.items():
                gain = ("" if not base or not base.ttft.p50 else
                        f"{(1 - s.ttft.p50 / base.ttft.p50) * 100:+.1f}%")
                rows.append({
                    "router": r, "TTFT p50": s.ttft.p50, "TTFT p95": s.ttft.p95,
                    "vs round_robin": gain,
                    "prefix hit %": s.prefix_hit_rate * 100,
                    "out tok/s": s.output_tok_per_s,
                    "goodput %": s.goodput_ratio * 100,
                })
            md += [report.md_table(rows), ""]
            report.bar_compare(
                list(per_router), {
                    "TTFT p50 (ms)": [s.ttft.p50 for s in per_router.values()],
                    "TTFT p95 (ms)": [s.ttft.p95 for s in per_router.values()],
                },
                out_dir / f"prefix-routing-{wname}-ttft.png",
                ylabel="ms", title=f"TTFT by routing strategy ({wname})", simulated=True)
            report.bar_compare(
                list(per_router),
                {"prefix hit rate (%)": [s.prefix_hit_rate * 100 for s in per_router.values()]},
                out_dir / f"prefix-routing-{wname}-hit.png",
                ylabel="%", title=f"Prefix cache hit rate ({wname})", simulated=True)

        md += ["## Reading this", "",
               "`unique` is the control: it has no shared prefix, so any routing",
               "strategy scoring better than round-robin there is measuring noise,",
               "not cache reuse.", ""]
        p = out_dir / "prefix-routing-report.md"
        p.write_text("\n".join(md))
        print(f"\nwrote {p}")
        return 0
    finally:
        for p in procs:
            p.stop()
        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
