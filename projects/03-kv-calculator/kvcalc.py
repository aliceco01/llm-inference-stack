#!/usr/bin/env python3
"""KV cache VRAM calculator: predict before you OOM.

    ./kvcalc.py plan     --model llama-3.1-8b --gpu h100-sxm --max-model-len 131072
    ./kvcalc.py capacity --model llama-3.1-70b --gpu h100-sxm --tp 4 --seq-len 8192
    ./kvcalc.py fit      --model llama-3.1-70b --gpu a100-80gb --seq-len 32768 --concurrency 32
    ./kvcalc.py table    --models llama-3.1-8b,qwen2.5-7b,llama-3.1-70b --gpu h100-sxm
    ./kvcalc.py tp-scan  --model llama-3.1-70b --gpu h100-sxm --seq-len 32768
    ./kvcalc.py config   --path ./my-model/config.json --gpu h100-sxm

Reads a real config.json with `--path` when the model is not in the registry,
which is the trustworthy route: a hardcoded table goes stale silently.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llmkit import ServingConfig, compute_budget, get_gpu, get_model, required_gpus
from llmkit.gpus import GPUS
from llmkit.modelspec import DTYPE_BYTES, REGISTRY, ModelSpec

GIB = 1024 ** 3


def load_model(args: argparse.Namespace) -> ModelSpec:
    if getattr(args, "path", None):
        return ModelSpec.from_hf_config(args.path, name=Path(args.path).parent.name)
    return get_model(args.model)


def cfg_from(args: argparse.Namespace) -> ServingConfig:
    return ServingConfig(
        tp=args.tp, pp=getattr(args, "pp", 1),
        weight_dtype=args.weight_dtype, kv_dtype=args.kv_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        max_model_len=getattr(args, "max_model_len", 8192),
        max_num_seqs=getattr(args, "max_num_seqs", 256),
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 8192),
        enable_cuda_graphs=not getattr(args, "no_cuda_graphs", False),
    )


def cmd_plan(args) -> int:
    m, g = load_model(args), get_gpu(args.gpu)
    cfg = cfg_from(args)
    b = compute_budget(m, g, cfg)
    print(m.describe())
    print(f"  source: {m.source}")
    print()
    print(b.explain())
    print()
    print("  Concurrency at various context lengths:")
    print(f"    {'seq len':>10}{'concurrent seqs':>18}{'KV per seq':>14}")
    for L in [512, 2048, 4096, 8192, 32768, 131072]:
        if cfg.max_model_len < L:
            continue
        n = b.max_concurrent_seqs(L)
        per = m.kv_bytes_for_sequence(L, cfg.kv_dtype) / GIB / max(cfg.tp, 1)
        print(f"    {L:>10,}{n:>18,}{per:>13.3f} GiB")
    if args.json:
        print("\n" + json.dumps({
            "model": m.name, "gpu": g.name,
            "kv_cache_gib": b.kv_cache_gib, "num_blocks": b.num_blocks,
            "max_cached_tokens": b.max_cached_tokens,
            "kv_bytes_per_token": b.kv_bytes_per_token,
            "fits": b.fits, "problems": b.problems,
        }, indent=2))
    return 0 if b.fits else 1


def cmd_capacity(args) -> int:
    m, g = load_model(args), get_gpu(args.gpu)
    cfg = cfg_from(args)
    cfg.max_model_len = max(cfg.max_model_len, args.seq_len)
    b = compute_budget(m, g, cfg)
    n = b.max_concurrent_seqs(args.seq_len)
    print(f"{m.name} on {cfg.tp}x {g.name}, seq_len={args.seq_len:,}")
    print(f"  KV cache            {b.kv_cache_gib:>10.2f} GiB")
    print(f"  KV per token        {b.kv_bytes_per_token/1024:>10.2f} KiB")
    print(f"  blocks per sequence {math.ceil(args.seq_len/cfg.block_size):>10,}")
    print(f"  max concurrent      {n:>10,} sequences")
    if args.concurrency:
        h = b.headroom_ratio(args.seq_len, args.concurrency)
        verdict = "FITS" if h >= 1.0 else "DOES NOT FIT"
        print(f"\n  target concurrency {args.concurrency}: {verdict} "
              f"(headroom {h:.2f}x)")
        if h < 1.0:
            print(f"    need {math.ceil(args.concurrency/max(n,1))}x more KV cache. Options:")
            print(f"      - raise TP to shard KV heads (limit: {m.num_kv_heads} KV heads)")
            print(f"      - kv_dtype fp8 halves KV: {n*2:,} sequences")
            print(f"      - cut max_model_len if requests do not need {args.seq_len:,}")
        return 0 if h >= 1.0 else 1
    return 0


def cmd_fit(args) -> int:
    m, g = load_model(args), get_gpu(args.gpu)
    res = required_gpus(m, g, target_seq_len=args.seq_len,
                        target_concurrency=args.concurrency, cfg=cfg_from(args),
                        max_tp=args.max_tp)
    print(f"{m.name} on {g.name}: need {args.concurrency} concurrent "
          f"sequences of {args.seq_len:,} tokens\n")
    print(f"  {'TP':>4}{'fits':>7}{'KV GiB':>10}{'max concurrent':>17}{'headroom':>11}")
    for a in res["attempts"]:
        print(f"  {a['tp']:>4}{('yes' if a['fits'] else 'no'):>7}{a['kv_gib']:>10.2f}"
              f"{a['max_concurrent']:>17,}{a['headroom']:>10.2f}x")
        for p in a["problems"]:
            print(f"        {p}")
    if res["ok"]:
        print(f"\n  ANSWER: TP={res['tp']} ({res['tp']}x {g.name})")
        return 0
    print(f"\n  ANSWER: does not fit on {g.name} at any TP up to {args.max_tp}.")
    print("  Consider: a larger-memory GPU, fp8 KV cache, or shorter max_model_len.")
    return 1


def cmd_tp_scan(args) -> int:
    """Show where TP stops buying KV cache.

    This is the counterintuitive one. Weights shard cleanly with TP, but KV
    heads only shard until TP reaches num_kv_heads. Past that they are
    replicated, so KV cache per GPU stops shrinking while you keep paying for
    GPUs.
    """
    m, g = load_model(args), get_gpu(args.gpu)
    print(f"{m.describe()}\non {g.name} (KV heads: {m.num_kv_heads})\n")
    print(f"  {'TP':>4}{'weights/GPU':>14}{'KV/tok/GPU':>13}{'KV GiB/GPU':>12}"
          f"{'total KV GiB':>14}{'seqs@' + str(args.seq_len):>14}")
    prev_kv_per_tok = None
    for tp in [1, 2, 4, 8, 16]:
        if tp > args.max_tp:
            break
        cfg = cfg_from(args); cfg.tp = tp
        cfg.max_model_len = max(cfg.max_model_len, args.seq_len)
        try:
            b = compute_budget(m, g, cfg)
        except Exception as e:
            print(f"  {tp:>4}  {e}")
            continue
        if b.kv_cache_gib <= 0:
            print(f"  {tp:>4}{b.weights_gib:>13.2f}G{b.kv_bytes_per_token/1024:>12.1f}K"
                  f"{'--':>12}{'--':>14}{'--':>14}  weights do not fit")
            prev_kv_per_tok = b.kv_bytes_per_token
            continue
        total = b.kv_cache_gib * tp
        n = b.max_concurrent_seqs(args.seq_len)
        flag = ""
        # The signal is per-token KV per GPU, not total KV. Total keeps rising
        # because sharded weights free room, which masks the fact that KV has
        # stopped sharding.
        if prev_kv_per_tok is not None and b.kv_bytes_per_token >= prev_kv_per_tok:
            flag = "  <-- KV/token stopped shrinking"
        if b.kv_heads_replicated:
            flag += " (KV heads replicated)"
        print(f"  {tp:>4}{b.weights_gib:>13.2f}G{b.kv_bytes_per_token/1024:>12.1f}K"
              f"{b.kv_cache_gib:>12.2f}{total:>14.2f}{n:>14,}{flag}")
        prev_kv_per_tok = b.kv_bytes_per_token
    return 0


def cmd_table(args) -> int:
    names = args.models.split(",") if args.models else list(REGISTRY)
    g = get_gpu(args.gpu)
    print(f"KV cache on 1x {g.name} ({g.vram_gb:.0f} GB), "
          f"weights {args.weight_dtype}, KV {args.kv_dtype}, TP={args.tp}\n")
    hdr = (f"{'model':<16}{'params':>8}{'KV KiB/tok':>12}{'weights G':>11}"
           f"{'KV GiB':>9}{'tokens':>10}{'@4k':>7}{'@32k':>7}{'@128k':>7}")
    print(hdr); print("-" * len(hdr))
    for n in names:
        try:
            m = get_model(n.strip())
        except KeyError as e:
            print(f"{n:<16} {e}"); continue
        cfg = cfg_from(args); cfg.max_model_len = 4096
        b = compute_budget(m, g, cfg)
        if not b.fits and b.kv_cache_gib <= 0:
            print(f"{m.name:<16}{m.params_b:>8.1f}{b.kv_bytes_per_token/1024:>12.1f}"
                  f"{b.weights_gib:>11.1f}{'-':>9}{'DOES NOT FIT':>10}")
            continue
        print(f"{m.name:<16}{m.params_b:>8.1f}{b.kv_bytes_per_token/1024:>12.1f}"
              f"{b.weights_gib:>11.1f}{b.kv_cache_gib:>9.1f}{b.max_cached_tokens:>10,}"
              f"{b.max_concurrent_seqs(4096):>7,}{b.max_concurrent_seqs(32768):>7,}"
              f"{b.max_concurrent_seqs(131072):>7,}")
    return 0


def cmd_config(args) -> int:
    m = ModelSpec.from_hf_config(args.path)
    print(m.describe())
    print(f"\n  attention kind   {m.attn_kind}")
    print(f"  KV bytes/token   fp16 {m.kv_bytes_per_token('fp16')/1024:.2f} KiB   "
          f"fp8 {m.kv_bytes_per_token('fp8')/1024:.2f} KiB")
    print(f"  estimated params {m.estimate_params()/1e9:.2f} B")
    if m.num_kv_heads != m.num_attention_heads:
        naive = 2 * m.num_layers * m.num_attention_heads * m.head_dim * 2 / 1024
        print(f"\n  NOTE: sizing KV off num_attention_heads instead of num_key_value_heads "
              f"would give {naive:.1f} KiB/token, a {m.gqa_ratio:.0f}x overestimate.")
    if args.gpu:
        print()
        print(compute_budget(m, get_gpu(args.gpu), cfg_from(args)).explain())
    return 0


def cmd_oom(args) -> int:
    """Will this workload OOM? Answer with the arithmetic, not a guess."""
    m, g = load_model(args), get_gpu(args.gpu)
    cfg = cfg_from(args)
    cfg.max_model_len = max(cfg.max_model_len, args.p95_seq_len)
    b = compute_budget(m, g, cfg)
    print(f"{m.name} on {cfg.tp}x {g.name}\n")
    print(f"  KV cache: {b.kv_cache_gib:.2f} GiB = {b.max_cached_tokens:,} tokens\n")
    rows = [("median", args.median_seq_len), ("p95", args.p95_seq_len),
            ("p99", args.p99_seq_len or args.p95_seq_len)]
    print(f"  {'case':>8}{'seq len':>10}{'x concurrency':>15}{'tokens needed':>16}"
          f"{'headroom':>11}{'verdict':>10}")
    worst_ok = True
    for name, L in rows:
        need = args.concurrency * math.ceil(L / cfg.block_size) * cfg.block_size
        head = b.max_cached_tokens / max(need, 1)
        ok = head >= 1.0
        worst_ok &= ok
        print(f"  {name:>8}{L:>10,}{args.concurrency:>15}{need:>16,}"
              f"{head:>10.2f}x{('OK' if ok else 'OOM'):>10}")
    print()
    if worst_ok:
        print("  VERDICT: fits, including the p99 case.")
        p95_need = args.concurrency * args.p95_seq_len
        print(f"  Headroom at p95 is {b.max_cached_tokens/max(p95_need,1):.2f}x. "
              "Below ~1.3x, expect preemption thrash under bursts.")
    else:
        print("  VERDICT: will OOM or preempt heavily. vLLM does not crash here, it")
        print("  preempts, so the symptom is a latency cliff and rising")
        print("  vllm:num_preemptions_total rather than an OOM traceback.")
    return 0 if worst_ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="llama-3.1-8b",
                       help=f"registry name ({', '.join(list(REGISTRY)[:4])}, ...)")
        p.add_argument("--path", default=None, help="path to a real config.json")
        p.add_argument("--gpu", default="h100-sxm",
                       help=f"one of: {', '.join(GPUS)}")
        p.add_argument("--tp", type=int, default=1)
        p.add_argument("--pp", type=int, default=1)
        p.add_argument("--weight-dtype", default="bf16", choices=list(DTYPE_BYTES))
        p.add_argument("--kv-dtype", default="fp16", choices=list(DTYPE_BYTES))
        p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--max-num-seqs", type=int, default=256)
        p.add_argument("--max-num-batched-tokens", type=int, default=8192)
        p.add_argument("--no-cuda-graphs", action="store_true")

    p = sub.add_parser("plan", help="full memory breakdown"); common(p)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("capacity", help="concurrent sequences at a length"); common(p)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=8192)

    p = sub.add_parser("fit", help="smallest TP that serves a target"); common(p)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--max-tp", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)

    p = sub.add_parser("tp-scan", help="where TP stops buying KV capacity"); common(p)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--max-tp", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)

    p = sub.add_parser("table", help="compare models"); common(p)
    p.add_argument("--models", default=None)

    p = sub.add_parser("config", help="inspect a config.json"); common(p)
    p.add_argument("--max-model-len", type=int, default=8192)

    p = sub.add_parser("oom", help="predict OOM for a workload"); common(p)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--median-seq-len", type=int, default=1024)
    p.add_argument("--p95-seq-len", type=int, default=8192)
    p.add_argument("--p99-seq-len", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=8192)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return {"plan": cmd_plan, "capacity": cmd_capacity, "fit": cmd_fit,
            "tp-scan": cmd_tp_scan, "table": cmd_table, "config": cmd_config,
            "oom": cmd_oom}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
