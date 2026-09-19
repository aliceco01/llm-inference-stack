#!/usr/bin/env python3
"""Quantization comparison lab: VRAM, latency and quality side by side.

    # rank candidates before spending a GPU-hour (no GPU needed)
    ./quantlab.py predict --model llama-3.1-70b --gpu h100-sxm --tp 2

    # emit the serve commands for each scheme
    ./quantlab.py plan --model meta-llama/Llama-3.1-8B-Instruct --gpu h100-sxm

    # full measured comparison against running endpoints (needs GPUs)
    ./quantlab.py run --config configs/llama8b-h100.yaml

`predict` is the cheap filter: it uses the roofline model to rank schemes so
you measure the two plausible ones instead of all six. `run` is the expensive
truth, and it refuses to report a latency win without a paired quality number,
because a latency win at unmeasured quality is not a result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from llmkit import get_gpu, get_model, report
from llmkit.quant import SCHEMES, compare


def cmd_predict(args) -> int:
    m, g = get_model(args.model), get_gpu(args.gpu)
    preds = compare(m, g, tp=args.tp, batch_size=args.batch_size,
                    ctx_len=args.ctx_len,
                    schemes=args.schemes.split(",") if args.schemes else None)
    print(f"{m.describe()}")
    print(f"on {args.tp}x {g.name}   (decode estimated at batch={args.batch_size}, "
          f"ctx={args.ctx_len})\n")
    hdr = (f"{'scheme':<14}{'weights':>9}{'KV cache':>10}{'KV tokens':>12}"
           f"{'decode':>9}{'prefill':>9}  notes")
    print(hdr); print("-" * 100)
    for p in preds:
        if not p.supported:
            print(f"{p.scheme:<14}{'--':>9}{'--':>10}{'--':>12}{'--':>9}{'--':>9}  "
                  f"UNSUPPORTED: {p.unsupported_reason}")
            continue
        print(f"{p.scheme:<14}{p.weights_gib:>8.1f}G{p.kv_cache_gib:>9.1f}G"
              f"{p.max_cached_tokens:>12,}{p.decode_speedup:>8.2f}x"
              f"{p.prefill_speedup:>8.2f}x  {p.notes[:44]}")
    print("\nrisks:")
    for p in preds:
        if p.supported and p.risks:
            print(f"  {p.scheme}:")
            for r in p.risks:
                print(f"    - {r}")
    print("\nReminder: these are PREDICTIONS from a roofline model, and none of "
          "them say anything about quality.\nUse `quality.py` to measure that, "
          "and treat any scheme you have not measured as unshippable.")
    if args.json:
        print("\n" + json.dumps([p.__dict__ for p in preds], indent=2, default=str))
    return 0


SERVE_TEMPLATE = """\
# {scheme}: {notes}
vllm serve {model} \\
  --port {port} \\
  --tensor-parallel-size {tp} \\
  --max-model-len {max_model_len} \\
  --gpu-memory-utilization 0.90 \\
  --enable-chunked-prefill --enable-prefix-caching \\
  {flags}
"""


def cmd_plan(args) -> int:
    """Emit the serve command for each scheme, with the checkpoint caveats."""
    g = get_gpu(args.gpu)
    names = args.schemes.split(",") if args.schemes else list(SCHEMES)
    port = args.base_port
    print(f"# Serving plan for {args.model} on {g.name}")
    print("#")
    print("# IMPORTANT: 4-bit schemes (AWQ, GPTQ) need a pre-quantized checkpoint.")
    print("# You cannot pass --quantization awq to an fp16 checkpoint and get AWQ;")
    print("# point --model at an already-quantized repo, or quantise offline first")
    print("# with llm-compressor / AutoAWQ. fp8 is the exception: vLLM can quantise")
    print("# weights to fp8 on load, which is part of why it is the easy default.")
    print()
    for n in names:
        s = SCHEMES.get(n)
        if not s:
            print(f"# unknown scheme {n}"); continue
        if not s.supported_on(g):
            print(f"# {s.name}: UNSUPPORTED on {g.name} "
                  f"(needs compute capability {s.requires_compute_capability})\n")
            continue
        model = args.model
        if s.weight_bits <= 4 and "awq" not in model.lower() and "gptq" not in model.lower():
            model = f"<pre-quantized-{s.name}-checkpoint>  # e.g. {args.model}-{s.name.upper()}"
        print(SERVE_TEMPLATE.format(
            scheme=s.name, notes=s.notes, model=model, port=port, tp=args.tp,
            max_model_len=args.max_model_len, flags=s.vllm_flag))
        port += 1
    return 0


async def _bench_one(name: str, url: str, model: str, args) -> dict[str, Any]:
    """Drive the project 02 harness against one endpoint."""
    from llmkit import (
        SLO,
        EndpointConfig,
        StreamingClient,
        WorkloadGenerator,
        run_closed_loop,
        summarize,
    )
    from llmkit.workload import LengthSpec, WorkloadSpec

    ep = EndpointConfig(base_url=url, model=model)
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)
    out: dict[str, Any] = {"scheme": name, "url": url, "points": []}
    # Two shapes, because weight-only quantization helps one and hurts the
    # other. Reporting a single blended number hides the whole tradeoff.
    shapes = {
        "decode_heavy": WorkloadSpec(name="decode_heavy",
                                     input_len=LengthSpec("fixed", 128),
                                     output_len=LengthSpec("fixed", 512)),
        "prefill_heavy": WorkloadSpec(name="prefill_heavy",
                                      input_len=LengthSpec("fixed", 4096),
                                      output_len=LengthSpec("fixed", 32)),
    }
    async with StreamingClient(ep) as client:
        for shape_name, spec in shapes.items():
            for c in args.concurrency:
                gen = WorkloadGenerator(spec)
                n = max(c * 8, 32)
                res = await run_closed_loop(client, gen.stream(n + c),
                                            concurrency=c, max_requests=n + c)
                s = summarize(res.records, label=f"{name}/{shape_name}",
                              concurrency=c, slo=slo, warmup_requests=c)
                out["points"].append({
                    "shape": shape_name, "concurrency": c,
                    "ttft_p50": s.ttft.p50, "ttft_p95": s.ttft.p95,
                    "itl_p50": s.itl.p50, "itl_p95": s.itl.p95,
                    "out_tok_s": s.output_tok_per_s,
                    "goodput": s.goodput_ratio,
                })
                print(f"    {shape_name:<14} c={c:<4} {s.headline()}")
    return out


async def cmd_run(args) -> int:
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.slo_ttft = cfg.get("slo", {}).get("ttft_ms", 2000)
    args.slo_itl = cfg.get("slo", {}).get("p_itl_ms", 100)
    args.concurrency = cfg.get("concurrency", [1, 8, 32])
    out_dir = Path(cfg.get("out", "results")); out_dir.mkdir(parents=True, exist_ok=True)

    baseline_name = cfg.get("baseline", "bf16")
    results: list[dict[str, Any]] = []
    for entry in cfg["schemes"]:
        print(f"\n=== {entry['name']} ({entry['url']}) ===")
        results.append(await _bench_one(entry["name"], entry["url"],
                                        entry.get("model", cfg["model"]), args))

    # --- quality, paired with every latency number ------------------------
    base = next((e for e in cfg["schemes"] if e["name"] == baseline_name), None)
    quality: dict[str, Any] = {}
    if base and not args.skip_quality:
        here = Path(__file__).resolve().parent
        for entry in cfg["schemes"]:
            if entry["name"] == baseline_name:
                continue
            print(f"\n=== quality: {entry['name']} vs {baseline_name} ===")
            qout = out_dir / f"quality-{entry['name']}.json"
            rc = subprocess.run(
                [sys.executable, str(here / "quality.py"),
                 "--baseline", base["url"], "--candidate", entry["url"],
                 "--model", entry.get("model", cfg["model"]),
                 "--n", str(cfg.get("quality_prompts", 32)),
                 "--out", str(qout)],
            ).returncode
            if rc == 0 and qout.exists():
                quality[entry["name"]] = json.loads(qout.read_text())

    # --- report -----------------------------------------------------------
    rows = []
    for r in results:
        for shape in ("decode_heavy", "prefill_heavy"):
            pts = [p for p in r["points"] if p["shape"] == shape]
            if not pts:
                continue
            best = max(pts, key=lambda p: p["out_tok_s"])
            q = quality.get(r["scheme"], {})
            rows.append({
                "scheme": r["scheme"], "shape": shape,
                "best tok/s": best["out_tok_s"],
                "TTFT p95": best["ttft_p95"], "ITL p95": best["itl_p95"],
                "exact match %": (q.get("agreement", {}).get("exact_match_rate", float("nan")) * 100
                                  if q else float("nan")),
                "task delta": (q.get("tasks", {}).get("delta", float("nan")) * 100
                               if q else float("nan")),
            })
    md = ["# Quantization comparison", "",
          f"- model: `{cfg['model']}`", f"- baseline: `{baseline_name}`",
          f"- concurrency levels: {args.concurrency}", "",
          report.md_table(rows), "",
          "## Reading this", "",
          "`decode_heavy` (128 in / 512 out) and `prefill_heavy` (4096 in / 32 out)",
          "are reported separately on purpose. Weight-only 4-bit schemes speed up",
          "the first and slow down the second, so a single blended throughput",
          "number would hide the entire tradeoff.", "",
          "A scheme with no quality column was not measured, and an unmeasured",
          "scheme is not a candidate.", ""]
    p = out_dir / "quantization-report.md"
    p.write_text("\n".join(md))
    print(f"\nwrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="roofline prediction, no GPU needed")
    p.add_argument("--model", default="llama-3.1-8b")
    p.add_argument("--gpu", default="h100-sxm")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--schemes", default=None)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("plan", help="emit serve commands per scheme")
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", default="h100-sxm")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--base-port", type=int, default=8000)
    p.add_argument("--schemes", default=None)

    p = sub.add_parser("run", help="measured comparison (needs running endpoints)")
    p.add_argument("--config", required=True)
    p.add_argument("--skip-quality", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    return {"predict": cmd_predict, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
