#!/usr/bin/env python3
"""Chunked prefill under mixed prefill/decode load.

    ./experiment.py policy   --out results/    # on vs off, the headline result
    ./experiment.py sweep    --out results/    # max_num_batched_tokens sweep
    ./experiment.py timeline --out results/    # per-step traces, both policies
    ./experiment.py real --base-url http://localhost:8000   # against live vLLM

The problem, stated precisely
-----------------------------
Without chunked prefill, vLLM's scheduler gives prefill absolute priority: if
any request is waiting and fits in the token budget, the whole step is a
prefill step and **every running sequence decodes nothing for its duration**.

A 32k-token prompt takes roughly 1.4s to prefill on an H100 for an 8B model.
Every user already mid-generation sees a 1.4 second gap between tokens. Their
TTFT was fine; their stream just froze because someone else submitted a long
document. That is decode starvation, and it is invisible in any benchmark that
sends uniform request shapes, which is most of them.

Chunked prefill splits a long prefill across steps and mixes it with decodes in
one batch. Decodes are admitted first, each costing one token of the budget, so
they cannot starve. Prefill consumes what remains.

The tradeoff is real and worth measuring rather than assuming: smaller chunks
protect ITL better and make TTFT worse, because a long prompt now needs more
steps to finish. `max_num_batched_tokens` is the dial.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from llmkit import SLO, report, summarize, sweep_table
from llmkit.metrics import RunSummary
from llmkit.simulator import EngineConfig, EngineSim, SimRequest


# ---------------------------------------------------------------------------
def mixed_workload(*, n_decode: int, n_long: int, long_len: int,
                   short_len: int, out_len: int, long_every_ms: float,
                   ) -> list[SimRequest]:
    """The load shape that exposes the problem.

    A population of ordinary short-prompt requests generating steadily, plus
    occasional long-document requests arriving during their generation. Uniform
    workloads cannot show decode starvation, because starvation is by
    definition one request class harming another.
    """
    reqs: list[SimRequest] = []
    # Steady interactive load, arriving early so they are mid-decode when the
    # long prompts land.
    for i in range(n_decode):
        reqs.append(SimRequest(
            request_id=f"chat-{i}", prompt_tokens=short_len,
            output_tokens=out_len, arrival_ms=i * 5.0,
        ))
    # Long prompts arriving later, spaced out.
    for j in range(n_long):
        reqs.append(SimRequest(
            request_id=f"doc-{j}", prompt_tokens=long_len,
            output_tokens=32, arrival_ms=250.0 + j * long_every_ms,
        ))
    return reqs


def run_sim(*, chunked: bool, max_batched: int, args,
            long_threshold: int = 0) -> tuple[EngineSim, list[Any]]:
    cfg = EngineConfig(
        model=args.model, gpu=args.gpu, tp=args.tp,
        max_model_len=max(args.long_len + 64, 4096),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max_batched,
        enable_chunked_prefill=chunked,
        long_prefill_token_threshold=long_threshold,
        enable_prefix_caching=False,   # isolate the scheduling effect
        seed=11,
    )
    e = EngineSim(cfg)
    e.submit(mixed_workload(
        n_decode=args.n_chat, n_long=args.n_long, long_len=args.long_len,
        short_len=args.short_len, out_len=args.out_len,
        long_every_ms=args.long_every_ms,
    ))
    e.run()
    return e, e.records()


def split_summaries(recs: list[Any], label: str, slo: SLO
                    ) -> tuple[RunSummary, RunSummary]:
    """Summarise the two request classes separately.

    Pooling them is what hides the effect: the long requests are few and the
    chat requests are many, so a pooled ITL percentile is dominated by chat
    requests whose stalls are exactly what we are trying to measure. Reporting
    per class is the whole point.
    """
    chat = [r for r in recs if r.request_id.startswith("chat")]
    docs = [r for r in recs if r.request_id.startswith("doc")]
    return (summarize(chat, label=f"{label} chat", slo=slo),
            summarize(docs, label=f"{label} doc", slo=slo))


def worst_stall_ms(recs: list[Any], prefix: str = "chat") -> float:
    """Longest single inter-token gap experienced by any request of a class.

    This is the number a user would describe as "it froze". A p99 ITL averages
    it away; the maximum is the complaint.
    """
    worst = 0.0
    for r in recs:
        if not r.request_id.startswith(prefix):
            continue
        itls = r.itls_ms
        if itls:
            worst = max(worst, max(itls))
    return worst


# ---------------------------------------------------------------------------
def cmd_policy(args) -> int:
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], {}
    for label, chunked in (("prefill-priority", False), ("chunked-prefill", True)):
        e, recs = run_sim(chunked=chunked, max_batched=args.max_batched, args=args)
        chat, doc = split_summaries(recs, label, slo)
        stall = worst_stall_ms(recs, "chat")
        summaries[label] = (chat, doc)
        st = e.stats()
        rows.append({
            "policy": label,
            "chat TTFT p50": chat.ttft.p50, "chat TTFT p95": chat.ttft.p95,
            "chat ITL p50": chat.itl.p50, "chat ITL p99": chat.itl.p99,
            "chat worst stall": stall,
            "doc TTFT p50": doc.ttft.p50,
            "out tok/s": chat.output_tok_per_s + doc.output_tok_per_s,
            "chat goodput %": chat.goodput_ratio * 100,
        })
        print(f"\n=== {label} ===")
        print(f"  {chat.headline()}")
        print(f"  {doc.headline()}")
        print(f"  worst single inter-token gap for a chat request: {stall:.1f} ms")
        print(f"  steps: {st['steps']}, time by phase: {st['time_by_phase_ms']}")

    print("\n" + report.md_table(rows))

    a, b = summaries["prefill-priority"], summaries["chunked-prefill"]
    stall_a = rows[0]["chat worst stall"]
    stall_b = rows[1]["chat worst stall"]
    print("\nInterpretation:")
    print(f"  chat worst stall  {stall_a:8.1f} ms -> {stall_b:8.1f} ms")
    print(f"  chat ITL p99      {a[0].itl.p99:8.2f} ms -> {b[0].itl.p99:8.2f} ms")
    print(f"  doc  TTFT p50     {a[1].ttft.p50:8.1f} ms -> {b[1].ttft.p50:8.1f} ms")
    print("  The stall improves because decodes are no longer blocked behind a")
    print("  whole prefill. Long-prompt TTFT usually worsens slightly, because")
    print("  its prefill now shares each step with decode work. That is the")
    print("  trade, and it is almost always worth taking.")

    report.bar_compare(
        ["chat ITL p50", "chat ITL p99", "chat worst stall"],
        {"prefill-priority": [a[0].itl.p50, a[0].itl.p99, stall_a],
         "chunked-prefill": [b[0].itl.p50, b[0].itl.p99, stall_b]},
        out / "chunked-prefill-itl.png", ylabel="ms", log=True,
        title="Decode latency for interactive requests, with long prompts in flight",
        simulated=True)
    report.bar_compare(
        ["chat TTFT p50", "chat TTFT p95", "doc TTFT p50"],
        {"prefill-priority": [a[0].ttft.p50, a[0].ttft.p95, a[1].ttft.p50],
         "chunked-prefill": [b[0].ttft.p50, b[0].ttft.p95, b[1].ttft.p50]},
        out / "chunked-prefill-ttft.png", ylabel="ms",
        title="TTFT: the other side of the trade", simulated=True)

    (out / "chunked-prefill-policy.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}/chunked-prefill-*.png and .json")
    return 0


def cmd_sweep(args) -> int:
    """max_num_batched_tokens is the TTFT/ITL dial. Show its shape."""
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")]
    rows = []
    series_chat: list[RunSummary] = []
    print(f"chunked prefill ON, sweeping max_num_batched_tokens "
          f"(long prompt = {args.long_len} tokens)\n")
    hdr = (f"{'budget':>8}{'chat ITL p50':>14}{'chat ITL p99':>14}"
           f"{'worst stall':>13}{'doc TTFT p50':>14}{'tok/s':>10}")
    print(hdr); print("-" * len(hdr))
    for b in budgets:
        e, recs = run_sim(chunked=True, max_batched=b, args=args)
        chat, doc = split_summaries(recs, f"b={b}", slo)
        chat.concurrency = b     # so the chart can use it as an x axis
        series_chat.append(chat)
        stall = worst_stall_ms(recs, "chat")
        print(f"{b:>8}{chat.itl.p50:>14.2f}{chat.itl.p99:>14.2f}"
              f"{stall:>13.1f}{doc.ttft.p50:>14.1f}"
              f"{chat.output_tok_per_s + doc.output_tok_per_s:>10.0f}")
        rows.append({"max_num_batched_tokens": b,
                     "chat ITL p50": chat.itl.p50, "chat ITL p99": chat.itl.p99,
                     "chat worst stall": stall,
                     "doc TTFT p50": doc.ttft.p50, "doc TTFT p95": doc.ttft.p95,
                     "total tok/s": chat.output_tok_per_s + doc.output_tok_per_s})
    print("\n" + report.md_table(rows))
    print("""
Reading the sweep: a smaller budget slices long prefills more finely, so
decodes wait less and ITL improves, while the long prompt needs more steps and
its TTFT worsens. The useful setting is the smallest budget that still keeps
prefill throughput acceptable. Setting it below roughly 512 starts to waste a
meaningful fraction of each step on fixed per-step overhead.""")
    report.latency_vs_load({"chat (interactive)": series_chat},
                           out / "chunked-prefill-budget-sweep.png",
                           slo=slo, x="concurrency",
                           title="Effect of max_num_batched_tokens "
                                 "(x axis is the token budget)",
                           simulated=True)
    (out / "chunked-prefill-sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}/chunked-prefill-budget-sweep.png")
    return 0


def cmd_timeline(args) -> int:
    """Per-step traces. This is the figure that makes the mechanism obvious."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for label, chunked in (("prefill-priority", False), ("chunked-prefill", True)):
        e, _ = run_sim(chunked=chunked, max_batched=args.max_batched, args=args)
        p = report.timeline(
            e.traces, out / f"timeline-{label}.png",
            title=f"Scheduler timeline: {label} "
                  f"(max_num_batched_tokens={args.max_batched})")
        phases: dict[str, float] = {}
        for t in e.traces:
            phases[t.phase] = phases.get(t.phase, 0.0) + t.dt_ms
        total = sum(phases.values()) or 1.0
        print(f"{label}:")
        for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<10} {v:8.1f} ms  ({v/total*100:5.1f}% of wall time)")
        longest = max((t.dt_ms for t in e.traces), default=0.0)
        print(f"  longest single step: {longest:.1f} ms "
              "(every decoding request stalls for this long)")
        print(f"  wrote {p}\n")
    print("In the prefill-priority timeline, decode steps are absent for long")
    print("stretches while prefill steps run: that gap is the stall. With chunked")
    print("prefill the steps become 'mixed' and decode never fully stops.")
    return 0


async def cmd_real(args) -> int:
    """Run the same experiment against a live endpoint.

    Start two vLLM instances, one with --enable-chunked-prefill and one with
    --no-enable-chunked-prefill, then point this at each in turn. The workload
    shape is what matters: interleaved short and long prompts.
    """
    from llmkit import EndpointConfig, Request, StreamingClient
    from llmkit.load import run_closed_loop

    ep = EndpointConfig(base_url=args.base_url, model=args.model)
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)

    from llmkit.workload import make_text

    def gen():
        i = 0
        while True:
            i += 1
            if i % args.long_every == 0:
                yield Request(prompt=make_text(args.long_len, seed=i),
                              max_tokens=32, request_id=f"doc-{i}",
                              prompt_tokens_hint=args.long_len)
            else:
                yield Request(prompt=make_text(args.short_len, seed=i),
                              max_tokens=args.out_len, request_id=f"chat-{i}",
                              prompt_tokens_hint=args.short_len)

    async with StreamingClient(ep) as client:
        res = await run_closed_loop(client, gen(), concurrency=args.concurrency,
                                    max_requests=args.requests)
    chat, doc = split_summaries(res.records, "live", slo)
    print(sweep_table([chat, doc]))
    print(f"\nworst single inter-token gap for a chat request: "
          f"{worst_stall_ms(res.records, 'chat'):.1f} ms")
    print("\nCompare this number between a server started with and without")
    print("--enable-chunked-prefill. It is the number that changes most.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="llama-3.1-8b")
        p.add_argument("--gpu", default="h100-sxm")
        p.add_argument("--tp", type=int, default=1)
        p.add_argument("--n-chat", type=int, default=24,
                       help="interactive requests generating throughout")
        p.add_argument("--n-long", type=int, default=4)
        p.add_argument("--long-len", type=int, default=16384)
        p.add_argument("--short-len", type=int, default=256)
        p.add_argument("--out-len", type=int, default=256)
        p.add_argument("--long-every-ms", type=float, default=600.0)
        p.add_argument("--max-num-seqs", type=int, default=64)
        p.add_argument("--max-batched", type=int, default=2048)
        p.add_argument("--slo-ttft", type=float, default=2000.0)
        p.add_argument("--slo-itl", type=float, default=50.0)
        p.add_argument("--out", default="results")

    p = sub.add_parser("policy", help="chunked prefill on vs off"); common(p)
    p = sub.add_parser("sweep", help="max_num_batched_tokens sweep"); common(p)
    p.add_argument("--budgets", default="256,512,1024,2048,4096,8192,16384")
    p = sub.add_parser("timeline", help="per-step traces"); common(p)
    p = sub.add_parser("real", help="against a live endpoint"); common(p)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--requests", type=int, default=160)
    p.add_argument("--long-every", type=int, default=8)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "real":
        return asyncio.run(cmd_real(args))
    return {"policy": cmd_policy, "sweep": cmd_sweep,
            "timeline": cmd_timeline}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
