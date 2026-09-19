#!/usr/bin/env python3
"""Live KV cache monitor for a running engine.

    ./kvmon.py --url http://localhost:8000 --interval 1
    ./kvmon.py --url http://localhost:8000 --csv kv.csv --duration 120
    ./kvmon.py --url http://localhost:8000 --predict --model llama-3.1-8b --gpu h100-sxm

Scrapes the Prometheus endpoint that vLLM and SGLang expose (and the project 01
simulator mimics), so the same tool watches a laptop simulation and a
production replica.

What it watches for, and why each one matters:

* `gpu_cache_usage_perc` climbing toward 1.0 is the leading indicator. Latency
  is still fine at 85%; by the time it is pinned at 100% you are already
  preempting.
* `num_preemptions_total` increasing at all is the alarm. Preemption means the
  engine ran out of KV blocks and threw away work it had already done. Under
  recompute mode that work gets redone, so throughput falls while GPU
  utilisation stays high, which is the confusing signature people misdiagnose
  as "the model got slower".
* `num_requests_waiting` growing while `num_requests_running` is flat means the
  batch is capped, by max_num_seqs or by memory. Which one it is decides whether
  the fix is a flag change or a bigger GPU.
* prefix cache hit rate falling is usually a routing regression, not an engine
  problem: project 04 exists to keep it high.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

from llmkit.prom import parse_prometheus


@dataclass
class Sample:
    t: float
    running: float = 0.0
    waiting: float = 0.0
    swapped: float = 0.0
    kv_util: float = 0.0
    preemptions: float = 0.0
    prompt_tokens: float = 0.0
    gen_tokens: float = 0.0
    prefix_queries: float = 0.0
    prefix_hits: float = 0.0
    blocks_total: float = 0.0
    blocks_used: float = 0.0
    blocks_evictable: float = 0.0

    @property
    def pinned_util(self) -> float:
        """Utilisation by blocks that CANNOT be reclaimed.

        With prefix caching on, a finished request's blocks stay resident so a
        later request with the same prefix hits. They count as used but are
        free on demand. Alerting on total utilisation therefore pages you at
        90% on an engine that is completely idle. The number that predicts
        preemption is the pinned fraction.
        """
        if not self.blocks_total:
            return self.kv_util
        return max(self.blocks_used - self.blocks_evictable, 0.0) / self.blocks_total

    @classmethod
    def of(cls, m: dict[str, float], t: float) -> Sample:
        def g(*names: str) -> float:
            for n in names:
                if n in m:
                    return m[n]
            return 0.0
        return cls(
            t=t,
            running=g("vllm:num_requests_running", "sglang:num_running_reqs"),
            waiting=g("vllm:num_requests_waiting", "sglang:num_queue_reqs"),
            swapped=g("vllm:num_requests_swapped"),
            kv_util=g("vllm:gpu_cache_usage_perc", "sglang:token_usage"),
            preemptions=g("vllm:num_preemptions_total"),
            prompt_tokens=g("vllm:prompt_tokens_total", "sglang:prompt_tokens_total"),
            gen_tokens=g("vllm:generation_tokens_total", "sglang:generation_tokens_total"),
            prefix_queries=g("vllm:prefix_cache_queries_total"),
            prefix_hits=g("vllm:prefix_cache_hits_total"),
            blocks_total=g("sim:num_gpu_blocks"),
            blocks_used=g("sim:num_gpu_blocks_used"),
            blocks_evictable=g("sim:num_gpu_blocks_evictable"),
        )


class Monitor:
    def __init__(self, url: str, *, window: int = 60) -> None:
        self.url = url.rstrip("/")
        self.history: deque[Sample] = deque(maxlen=window)
        self.alerts: list[str] = []
        self._fired: set[str] = set()

    def scrape(self) -> Sample | None:
        try:
            r = httpx.get(f"{self.url}/metrics", timeout=5.0)
            r.raise_for_status()
        except Exception as e:
            print(f"  scrape failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None
        s = Sample.of(parse_prometheus(r.text), time.time())
        self.history.append(s)
        self._check(s)
        return s

    def rate(self, name: str) -> float:
        """Per-second rate of a counter over the retained window."""
        if len(self.history) < 2:
            return 0.0
        a, b = self.history[0], self.history[-1]
        dt = b.t - a.t
        if dt <= 0:
            return 0.0
        return max(getattr(b, name) - getattr(a, name), 0.0) / dt

    def _check(self, s: Sample) -> None:
        def fire(key: str, msg: str) -> None:
            if key not in self._fired:
                self._fired.add(key)
                self.alerts.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        # Alert on PINNED utilisation, not total. See Sample.pinned_util.
        u = s.pinned_util
        if u >= 0.95:
            fire("kv95", f"pinned KV at {u*100:.1f}%: preemption is imminent. "
                         "Reduce max_num_seqs, shorten max_model_len, or add capacity.")
        elif u >= 0.85:
            fire("kv85", f"pinned KV at {u*100:.1f}%: headroom is thin. "
                         "A burst of long prompts will preempt.")
        if len(self.history) >= 2 and s.preemptions > self.history[-2].preemptions:
            fire("preempt", f"PREEMPTION: {s.preemptions:.0f} total. The engine is "
                            "discarding computed KV and redoing work. Throughput "
                            "falls while GPU utilisation stays high.")
        if s.waiting > 0 and len(self.history) >= 10:
            older = self.history[-10]
            if s.waiting > older.waiting * 2 and s.running <= older.running:
                fire("queue", f"queue growing ({older.waiting:.0f} -> {s.waiting:.0f}) "
                              "while the running batch is flat: arrival rate exceeds "
                              "service rate. This diverges, it does not recover.")
        if s.swapped > 0:
            fire("swap", f"{s.swapped:.0f} requests swapped to CPU: KV cache is "
                         "oversubscribed and PCIe is now in the latency path.")

    def render(self, s: Sample, budget: Any = None) -> str:
        hit = (s.prefix_hits / s.prefix_queries * 100) if s.prefix_queries else float("nan")
        bar_w = 34
        filled = int(min(max(s.kv_util, 0.0), 1.0) * bar_w)
        colour = "\033[31m" if s.kv_util >= 0.95 else ("\033[33m" if s.kv_util >= 0.85 else "\033[32m")
        bar = f"{colour}{'#' * filled}\033[0m{'.' * (bar_w - filled)}"
        pin_filled = int(min(max(s.pinned_util, 0.0), 1.0) * bar_w)
        lines = [
            f"  KV cache  [{bar}] {s.kv_util*100:5.1f}% total",
            f"    pinned  [{'=' * pin_filled}{'.' * (bar_w - pin_filled)}] "
            f"{s.pinned_util*100:5.1f}% (rest is reclaimable prefix cache)",
            f"  requests   running {s.running:>5.0f}   waiting {s.waiting:>5.0f}"
            f"   swapped {s.swapped:>4.0f}",
            f"  tokens/s   prompt {self.rate('prompt_tokens'):>8.1f}"
            f"   generated {self.rate('gen_tokens'):>8.1f}",
            f"  prefix     hit rate {hit:>5.1f}%"
            f"   ({s.prefix_hits:,.0f} / {s.prefix_queries:,.0f} tokens)",
            f"  preemptions {s.preemptions:>4.0f} total"
            f"   ({self.rate('preemptions')*60:.1f}/min)",
        ]
        if s.blocks_total:
            lines.append(
                f"  blocks     {s.blocks_used:,.0f} / {s.blocks_total:,.0f} used"
                f"   ({s.blocks_evictable:,.0f} evictable)")
        if budget is not None and s.blocks_total == 0 and budget.num_blocks:
            est_tokens = s.kv_util * budget.max_cached_tokens
            lines.append(f"  estimated  {est_tokens:,.0f} / {budget.max_cached_tokens:,.0f} "
                         f"KV tokens in use (predicted budget)")
        return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until interrupted")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--predict", action="store_true",
                    help="also show the predicted budget from kvcalc")
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--gpu", default="h100-sxm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    budget = None
    if args.predict:
        from llmkit import ServingConfig, compute_budget, get_gpu, get_model
        budget = compute_budget(
            get_model(args.model), get_gpu(args.gpu),
            ServingConfig(tp=args.tp, max_model_len=args.max_model_len),
        )
        print(budget.explain()); print()

    mon = Monitor(args.url)
    writer = None
    fh = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t", "running", "waiting", "swapped", "kv_util",
                         "pinned_util", "preemptions", "prompt_tokens", "gen_tokens",
                         "prefix_queries", "prefix_hits"])

    t_end = time.time() + args.duration if args.duration else None
    n_alerts = 0
    try:
        while True:
            s = mon.scrape()
            if s:
                if not args.once:
                    print("\033[2J\033[H", end="")
                print(f"KV monitor  {args.url}   {time.strftime('%H:%M:%S')}")
                print(mon.render(s, budget))
                if mon.alerts:
                    print("\n  alerts:")
                    for a in mon.alerts[-8:]:
                        print(f"    {a}")
                    n_alerts = len(mon.alerts)
                if writer:
                    writer.writerow([s.t, s.running, s.waiting, s.swapped, s.kv_util,
                                     s.pinned_util, s.preemptions, s.prompt_tokens,
                                     s.gen_tokens, s.prefix_queries, s.prefix_hits])
                    fh.flush()
            if args.once:
                break
            if t_end and time.time() >= t_end:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if fh:
            fh.close()
            print(f"wrote {args.csv}")
    return 1 if n_alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
