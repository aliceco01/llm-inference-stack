#!/usr/bin/env python3
"""Disaggregated prefill/decode vs colocated serving.

    ./experiment.py split    --gpus 8 --input-len 4096 --output-len 512
    ./experiment.py compare  --gpus 8 --out results/
    ./experiment.py sweep    --out results/     # where does disaggregation win?
    ./experiment.py transfer --out results/     # the KV transfer cost model

`split` is the planning question (how many GPUs in each pool), `compare` runs
both topologies over an identical request stream, and `sweep` maps the region
of workload space where disaggregation actually helps, which is smaller than
the architecture's popularity suggests.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llmkit import get_model, report
from llmkit.cluster import (
    INTERCONNECTS,
    ClusterConfig,
    ClusterRequest,
    ClusterSim,
    optimal_split,
)


def make_requests(n: int, *, input_len: int, output_len: int, rps: float,
                  seed: int = 3, jitter: float = 0.3) -> list[ClusterRequest]:
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rps) * 1000.0
        il = max(16, int(input_len * (1 + rng.uniform(-jitter, jitter))))
        ol = max(8, int(output_len * (1 + rng.uniform(-jitter, jitter))))
        reqs.append(ClusterRequest(f"r{i}", il, ol, arrival_ms=t))
    return reqs


def cmd_split(args) -> int:
    r = optimal_split(args.model, args.gpu, total_gpus=args.gpus,
                      input_len=args.input_len, output_len=args.output_len)
    print(f"{args.model} on {args.gpus}x {args.gpu}, "
          f"{args.input_len} in / {args.output_len} out\n")
    for k, v in r.items():
        if k == "verdict":
            continue
        print(f"  {k:<32} {v}")
    print(f"\n  {r['verdict']}")
    print("""
  How to read the split: prefill work per request scales with input length,
  decode work scales with output length times per-step cost at the achieved
  batch size. The pool ratio should match the ratio of those totals, which
  means the right split MOVES with your traffic. A fixed 50/50 is almost never
  correct, and a split tuned for summarisation traffic is wrong for chat.""")
    return 0


def cmd_compare(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    reqs_template = make_requests(args.requests, input_len=args.input_len,
                                  output_len=args.output_len, rps=args.rps)

    def fresh():
        return [ClusterRequest(r.request_id, r.prompt_tokens, r.output_tokens,
                               r.arrival_ms) for r in reqs_template]

    plan = optimal_split(args.model, args.gpu, total_gpus=args.gpus,
                         input_len=args.input_len, output_len=args.output_len)
    np_, nd = plan["suggested_prefill_gpus"], plan["suggested_decode_gpus"]

    rows = []
    # colocated baseline
    cfg = ClusterConfig(model=args.model, prefill_gpu=args.gpu, decode_gpu=args.gpu,
                        interconnect=args.interconnect,
                        chunked_prefill=not args.no_chunked_prefill)
    sim = ClusterSim(cfg)
    colo = sim.run_colocated(fresh(), args.gpus)
    rows.append(colo.summary())
    print(f"colocated ({args.gpus} workers, chunked_prefill="
          f"{not args.no_chunked_prefill}):")
    print(f"  {json.dumps(colo.summary(), indent=None)}")

    # disaggregated at the suggested split, plus neighbours
    candidates = {(np_, nd)}
    for d in (-2, -1, 1, 2):
        p = np_ + d
        if 1 <= p < args.gpus:
            candidates.add((p, args.gpus - p))
    for p, d in sorted(candidates):
        cfg = ClusterConfig(model=args.model, prefill_gpu=args.gpu,
                            decode_gpu=args.gpu, n_prefill=p, n_decode=d,
                            interconnect=args.interconnect)
        sim = ClusterSim(cfg)
        dis = sim.run_disaggregated(fresh())
        s = dis.summary()
        s["topology"] = f"disagg {p}P/{d}D"
        rows.append(s)
        mark = "  <- suggested" if (p, d) == (np_, nd) else ""
        print(f"disaggregated {p}P/{d}D:{mark}")
        print(f"  {json.dumps(s, indent=None)}")
        for n in dis.notes:
            print(f"    NOTE: {n}")

    print("\n" + report.md_table(rows, columns=[
        "topology", "ttft_p50", "ttft_p95", "e2e_p95", "out_tok_per_s",
        "prefill_util", "decode_util", "transfer_ms_total"]))

    best = max(rows[1:], key=lambda r: r["out_tok_per_s"])
    base = rows[0]
    print(f"\nbest disaggregated: {best['topology']}")
    print(f"  throughput {base['out_tok_per_s']:.0f} -> {best['out_tok_per_s']:.0f} tok/s "
          f"({best['out_tok_per_s']/max(base['out_tok_per_s'],1e-9):.2f}x)")
    print(f"  TTFT p95   {base['ttft_p95']:.0f} -> {best['ttft_p95']:.0f} ms")
    print(f"  KV transfer added {best['transfer_ms_total']:.0f} ms of link time total")
    print("\nSIMULATED. These are roofline-model timings, not hardware measurements.")

    report.bar_compare(
        [r["topology"] for r in rows],
        {"TTFT p95 (ms)": [r["ttft_p95"] for r in rows],
         "E2E p95 (ms)": [r["e2e_p95"] for r in rows]},
        out / "disagg-latency.png", ylabel="ms", log=True,
        title="Latency: colocated vs disaggregated splits", simulated=True)
    report.bar_compare(
        [r["topology"] for r in rows],
        {"output tok/s": [r["out_tok_per_s"] for r in rows]},
        out / "disagg-throughput.png", ylabel="tok/s",
        title="Throughput by topology", simulated=True)
    (out / "disagg-compare.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/disagg-*.png")
    return 0


def cmd_sweep(args) -> int:
    """Map the workload region where disaggregation wins."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    shapes = [
        ("short chat", 256, 128),
        ("chat", 1024, 256),
        ("balanced", 2048, 512),
        ("RAG", 8192, 256),
        ("summarise", 16384, 512),
        ("long gen", 512, 2048),
        ("long doc + long gen", 16384, 2048),
    ]
    print(f"{args.gpus} GPUs total, {args.model} on {args.gpu}, "
          f"link={args.interconnect}\n")
    hdr = (f"{'workload':<22}{'in':>7}{'out':>7}{'colo tok/s':>12}"
           f"{'disagg tok/s':>14}{'gain':>8}{'xfer/prefill':>14}{'verdict':>10}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, il, ol in shapes:
        reqs = make_requests(args.requests, input_len=il, output_len=ol, rps=args.rps)

        # `reqs` is bound as a default so the closure captures THIS iteration's
        # requests. Without it the function would read whatever `reqs` holds at
        # call time, which is correct only by accident of call ordering.
        def fresh(reqs=reqs):
            return [ClusterRequest(r.request_id, r.prompt_tokens,
                                   r.output_tokens, r.arrival_ms) for r in reqs]

        cfg = ClusterConfig(model=args.model, prefill_gpu=args.gpu,
                            decode_gpu=args.gpu, interconnect=args.interconnect)
        colo = ClusterSim(cfg).run_colocated(fresh(), args.gpus)

        plan = optimal_split(args.model, args.gpu, total_gpus=args.gpus,
                             input_len=il, output_len=ol)
        p, d = plan["suggested_prefill_gpus"], plan["suggested_decode_gpus"]
        cfg2 = ClusterConfig(model=args.model, prefill_gpu=args.gpu,
                             decode_gpu=args.gpu, n_prefill=p, n_decode=d,
                             interconnect=args.interconnect)
        dis = ClusterSim(cfg2).run_disaggregated(fresh())

        c_tps, d_tps = colo.output_tok_per_s(), dis.output_tok_per_s()
        gain = d_tps / max(c_tps, 1e-9)
        xfer_pct = plan["transfer_as_pct_of_prefill"]
        verdict = "disagg" if gain > 1.05 else ("colo" if gain < 0.95 else "tie")
        print(f"{name:<22}{il:>7}{ol:>7}{c_tps:>12.0f}{d_tps:>14.0f}"
              f"{gain:>7.2f}x{xfer_pct:>13.1f}%{verdict:>10}")
        rows.append({"workload": name, "input": il, "output": ol,
                     "split": f"{p}P/{d}D",
                     "colocated tok/s": c_tps, "disagg tok/s": d_tps,
                     "gain": gain, "transfer % of prefill": xfer_pct,
                     "verdict": verdict})
    print("\n" + report.md_table(rows))
    print("""
The pattern to look for: disaggregation wins where prefill and decode demands
are IMBALANCED and large enough that interference costs real time, and loses
where prompts are short (transfer dominates a cheap prefill) or where the two
phases are already well matched inside one replica.

It is not a universal upgrade. It is a fleet-shaping tool that pays off at
scale and on skewed workloads, which is exactly why frontier serving stacks use
it and a two-GPU deployment usually should not.""")
    (out / "disagg-sweep.json").write_text(json.dumps(rows, indent=2))
    return 0


def cmd_transfer(args) -> int:
    """KV transfer cost: the thing that decides whether any of this works."""
    m = get_model(args.model)
    print(f"{m.describe()}\nKV per token: "
          f"{m.kv_bytes_per_token(args.kv_dtype)/1024:.1f} KiB ({args.kv_dtype})\n")
    lens = [512, 2048, 8192, 32768, 131072]
    hdr = f"{'prompt tokens':>15}{'KV size':>12}" + "".join(
        f"{k:>14}" for k in INTERCONNECTS)
    print(hdr); print("-" * len(hdr))
    for L in lens:
        nbytes = L * m.kv_bytes_per_token(args.kv_dtype)
        row = f"{L:>15,}{nbytes/1024**2:>10.0f}MB"
        for name, link in INTERCONNECTS.items():
            row += f"{link.transfer_ms(nbytes):>12.1f}ms"
        print(row)
    print("""
Compare these against prefill time for the same prompt on the same model:
a transfer that costs a large fraction of the prefill it is offloading has
eaten the benefit before decode even starts.

Three mitigations used in production stacks:
  * **Layer-wise streaming.** Send each layer's KV as it is computed rather
    than the whole cache at the end, overlapping transfer with prefill. This is
    what makes disaggregation viable at long context.
  * **fp8 KV cache.** Halves the bytes on the wire as well as in memory, which
    is a second reason project 05 lists kv-only quantization separately.
  * **Prefix-aware placement.** If the decode worker already holds the prefix,
    only the suffix needs transferring. Project 04's routing applies here too.

Note that a single node with NVLink between prefill and decode GPUs changes the
arithmetic completely: at 400 GB/s the transfer is near-free, which is why
intra-node disaggregation is much easier to justify than cross-node.""")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="llama-3.1-8b")
        p.add_argument("--gpu", default="h100-sxm")
        p.add_argument("--gpus", type=int, default=8)
        p.add_argument("--input-len", type=int, default=4096)
        p.add_argument("--output-len", type=int, default=512)
        p.add_argument("--requests", type=int, default=200)
        p.add_argument("--rps", type=float, default=8.0)
        p.add_argument("--interconnect", default="rdma-400g",
                       choices=list(INTERCONNECTS))
        p.add_argument("--out", default="results")

    p = sub.add_parser("split", help="suggested pool ratio"); common(p)
    p = sub.add_parser("compare", help="colocated vs disaggregated"); common(p)
    p.add_argument("--no-chunked-prefill", action="store_true")
    p = sub.add_parser("sweep", help="where disaggregation wins"); common(p)
    p = sub.add_parser("transfer", help="KV transfer cost model"); common(p)
    p.add_argument("--kv-dtype", default="fp16")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return {"split": cmd_split, "compare": cmd_compare, "sweep": cmd_sweep,
            "transfer": cmd_transfer}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
