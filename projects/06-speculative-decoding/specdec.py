#!/usr/bin/env python3
"""Speculative decoding: plan it, simulate it, and track acceptance in production.

    # which draft strategy, what k, how much speedup
    ./specdec.py plan --batch-size 1
    ./specdec.py plan --batch-size 64        # watch the gain collapse

    # sensitivity of speedup to acceptance rate and draft length
    ./specdec.py curve --draft-cost 0.16 --out results/

    # simulate end to end (no GPU)
    ./specdec.py simulate --alpha 0.72 --k 4 --model llama-3.1-8b --gpu h100-sxm

    # live acceptance tracking against a vLLM endpoint running speculation
    ./specdec.py watch --url http://localhost:8000 --interval 2

    # emit vLLM serve commands
    ./specdec.py serve-cmd --target meta-llama/Llama-3.1-70B-Instruct
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import httpx

from llmkit.prom import parse_prometheus
from llmkit.specdec import breakeven_alpha, expected_tokens, plan, speedup


def cmd_plan(args) -> int:
    plans = plan(batch_size=args.batch_size, k_max=args.k_max,
                 alpha_override=args.alpha)
    print(f"Draft strategies at batch size {args.batch_size}"
          + (f", forced alpha={args.alpha}" if args.alpha else "") + "\n")
    hdr = f"{'strategy':<24}{'alpha':>7}{'best k':>8}{'speedup':>9}{'breakeven a':>13}{'+VRAM':>8}"
    print(hdr); print("-" * len(hdr))
    for p in plans:
        print(f"{p.candidate:<24}{p.alpha:>7.2f}{p.k:>8}{p.speedup:>8.2f}x"
              f"{p.breakeven_alpha:>13.2f}{p.extra_vram_gib:>7.1f}G")
    print()
    for p in plans:
        if p.warnings:
            print(f"{p.candidate}:")
            for w in p.warnings:
                print(f"  - {w}")
    print("\nNote: acceptance rates here are representative starting points, not")
    print("measurements. They are strongly workload dependent. Measure yours with")
    print("`./specdec.py watch` before committing to a configuration.")
    return 0


def cmd_curve(args) -> int:
    """Speedup as a function of acceptance rate and draft length."""
    alphas = [i / 20 for i in range(1, 20)]
    ks = [1, 2, 3, 4, 6, 8, 12]
    print(f"Speedup vs acceptance rate (draft cost ratio {args.draft_cost}, "
          f"verify overhead {args.verify_overhead})\n")
    print(f"{'alpha':>7}" + "".join(f"{'k=' + str(k):>9}" for k in ks))
    print("-" * (7 + 9 * len(ks)))
    for a in alphas:
        row = f"{a:>7.2f}"
        for k in ks:
            s = speedup(a, k, args.draft_cost, args.verify_overhead)
            row += f"{s:>9.2f}"
        print(row)
    print("\nbreakeven acceptance rate by k:")
    for k in ks:
        print(f"  k={k:<3} alpha must exceed {breakeven_alpha(k, args.draft_cost, args.verify_overhead):.3f}")
    print("\nNaive vs correct expected-token counts (the k*alpha error):")
    print(f"  {'alpha':>7}{'k':>4}{'k*alpha (wrong)':>18}{'correct':>10}{'overstated by':>15}")
    for a in (0.5, 0.7, 0.9):
        for k in (4, 8):
            naive = k * a
            real = expected_tokens(a, k) - 1
            print(f"  {a:>7.2f}{k:>4}{naive:>18.2f}{real:>10.2f}"
                  f"{(naive / real - 1) * 100:>14.0f}%")

    if args.out:
        try:
            from llmkit import report
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
            for i, k in enumerate(ks):
                ax.plot(alphas, [speedup(a, k, args.draft_cost, args.verify_overhead)
                                 for a in alphas],
                        label=f"k={k}", color=report.PALETTE[i % len(report.PALETTE)])
            ax.axhline(1.0, color="#D55E00", ls=":", lw=1.4, label="breakeven")
            ax.set_xlabel("per-token acceptance rate (alpha)")
            ax.set_ylabel("decode speedup")
            ax.set_title(f"Speculative decoding speedup (draft cost {args.draft_cost})")
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)
            p = out / "specdec-speedup-curve.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            print(f"\nwrote {p}")
        except Exception as e:
            print(f"\nchart skipped: {type(e).__name__}: {e}")
    return 0


def cmd_simulate(args) -> int:
    """End-to-end simulation with and without speculation."""
    from llmkit.metrics import summarize, sweep_table
    from llmkit.simulator import EngineConfig, EngineSim, SimRequest
    from llmkit.types import SLO

    rows = []
    for label, k, alpha in (("baseline", 0, 0.0), (f"spec k={args.k}", args.k, args.alpha)):
        cfg = EngineConfig(
            model=args.model, gpu=args.gpu, max_model_len=args.max_model_len,
            spec_draft_tokens=k, spec_acceptance_rate=alpha,
            spec_draft_cost_ratio=args.draft_cost, seed=7,
        )
        e = EngineSim(cfg)
        e.submit([SimRequest(f"r{i}", prompt_tokens=args.input_len,
                             output_tokens=args.output_len, arrival_ms=0.0)
                  for i in range(args.concurrency)])
        e.run()
        s = summarize(e.records(), label=label, concurrency=args.concurrency,
                      slo=SLO(ttft_ms=5000, p_itl_ms=200))
        rows.append(s)
        st = e.stats()
        print(f"{label:<14} acceptance measured: {st.get('spec_acceptance')}")
    print()
    print(sweep_table(rows))
    base, spec = rows[0], rows[1]
    if base.itl.p50 and spec.itl.p50:
        print(f"\nITL p50: {base.itl.p50:.2f}ms -> {spec.itl.p50:.2f}ms "
              f"({base.itl.p50 / spec.itl.p50:.2f}x faster)")
    print("\nSIMULATED: timings come from the roofline model, not hardware.")
    return 0


SPEC_METRICS = [
    "vllm:spec_decode_draft_acceptance_rate",
    "vllm:spec_decode_efficiency",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_emitted_tokens_total",
]


def cmd_watch(args) -> int:
    """Track acceptance rate on a live endpoint.

    Acceptance is not a constant. It depends on the draft/target pair, on
    temperature, and heavily on the traffic mix: prompt-lookup speculation is
    excellent on summarisation and useless on open-ended chat. A deploy that
    was a win at launch becomes a loss when the traffic shifts, silently, so
    this is a metric to alert on and not just to measure once.
    """
    from llmkit.metrics import percentile
    prev: dict[str, float] = {}
    accepted_hist: list[float] = []
    print(f"watching {args.url} (ctrl-c to stop)\n")
    try:
        while True:
            try:
                r = httpx.get(f"{args.url.rstrip('/')}/metrics", timeout=5.0)
                r.raise_for_status()
            except Exception as e:
                print(f"scrape failed: {type(e).__name__}: {e}")
                time.sleep(args.interval); continue
            vals = parse_prometheus(r.text)
            have = {k: v for k, v in vals.items() if k.startswith("vllm:spec_decode")}
            if not have:
                print("no vllm:spec_decode_* metrics on this endpoint.")
                print("Speculative decoding is probably not enabled. Start vLLM with")
                print("  --speculative-config '{\"model\": \"...\", \"num_speculative_tokens\": 4}'")
                return 2
            draft = vals.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
            acc = vals.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
            emit = vals.get("vllm:spec_decode_num_emitted_tokens_total", 0.0)
            d_draft = draft - prev.get("draft", draft)
            d_acc = acc - prev.get("acc", acc)
            d_emit = emit - prev.get("emit", emit)
            prev = {"draft": draft, "acc": acc, "emit": emit}
            alpha = d_acc / d_draft if d_draft else float("nan")
            if not math.isnan(alpha):
                accepted_hist.append(alpha)
            # tokens emitted per target forward = emitted / (draft/k + ...) is
            # not directly available, so report the ratio vLLM reports instead.
            eff = vals.get("vllm:spec_decode_efficiency", float("nan"))
            cum = acc / draft if draft else float("nan")
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"alpha(window) {alpha:6.3f}   alpha(cumulative) {cum:6.3f}   "
                  f"efficiency {eff:6.3f}   emitted/s {d_emit / args.interval:8.1f}")
            if accepted_hist and len(accepted_hist) >= 5:
                be = breakeven_alpha(args.k, args.draft_cost)
                p05 = percentile(accepted_hist, 5)
                if p05 < be:
                    print(f"    WARNING: p5 acceptance {p05:.3f} is below breakeven "
                          f"{be:.3f} for k={args.k}. Speculation is losing on part "
                          "of your traffic. Consider lowering k or disabling it.")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if accepted_hist:
            print(f"\nacceptance over {len(accepted_hist)} samples: "
                  f"p5 {percentile(accepted_hist,5):.3f}  "
                  f"p50 {percentile(accepted_hist,50):.3f}  "
                  f"p95 {percentile(accepted_hist,95):.3f}")
    return 0


def cmd_serve_cmd(args) -> int:
    print(f"""# Speculative decoding configurations for {args.target}
#
# 1) Draft model. The draft MUST share the target's tokenizer/vocabulary.
#    A draft from another family cannot be used at all.
vllm serve {args.target} \\
  --tensor-parallel-size {args.tp} \\
  --speculative-config '{{"model": "{args.draft}", "num_speculative_tokens": {args.k}}}' \\
  --max-model-len {args.max_model_len} \\
  --port 8000

# 2) N-gram / prompt lookup. No draft model, no extra VRAM, no checkpoint.
#    Proposes continuations copied from the prompt itself. Try this FIRST:
#    it is free, and on summarisation/RAG/code-edit traffic it is often
#    competitive with a real draft model.
vllm serve {args.target} \\
  --tensor-parallel-size {args.tp} \\
  --speculative-config '{{"method": "ngram", "num_speculative_tokens": {args.k}, \\
                          "prompt_lookup_max": 4, "prompt_lookup_min": 2}}' \\
  --port 8001

# 3) EAGLE head. Highest acceptance per unit cost, but requires a head
#    trained against this exact target checkpoint.
vllm serve {args.target} \\
  --tensor-parallel-size {args.tp} \\
  --speculative-config '{{"method": "eagle", "model": "<eagle-head-for-this-target>", \\
                          "num_speculative_tokens": {args.k}}}' \\
  --port 8002

# Then measure, do not assume:
#   ./specdec.py watch --url http://localhost:8000 --k {args.k}
#   ../02-benchmark-suite/bench.py sweep --base-url http://localhost:8000 \\
#       --concurrency 1,4,16,64 --model {args.target}
#
# Expect the benefit to shrink as concurrency rises. Speculation trades FLOPs
# for bandwidth, and that trade stops paying once the batch is compute-bound.""")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="rank draft strategies")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--alpha", type=float, default=None)

    p = sub.add_parser("curve", help="speedup vs acceptance rate")
    p.add_argument("--draft-cost", type=float, default=0.16)
    p.add_argument("--verify-overhead", type=float, default=0.0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("simulate", help="end-to-end simulation")
    p.add_argument("--alpha", type=float, default=0.72)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--draft-cost", type=float, default=0.16)
    p.add_argument("--model", default="llama-3.1-8b")
    p.add_argument("--gpu", default="h100-sxm")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)

    p = sub.add_parser("watch", help="live acceptance tracking")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--draft-cost", type=float, default=0.16)

    p = sub.add_parser("serve-cmd", help="emit vLLM commands")
    p.add_argument("--target", default="meta-llama/Llama-3.1-70B-Instruct")
    p.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return {"plan": cmd_plan, "curve": cmd_curve, "simulate": cmd_simulate,
            "watch": cmd_watch, "serve-cmd": cmd_serve_cmd}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
