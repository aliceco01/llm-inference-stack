#!/usr/bin/env python3
"""Benchmark the fused kernels against PyTorch. Requires CUDA.

    ./bench.py rmsnorm --hidden 4096
    ./bench.py softmax --cols 4096
    ./bench.py all --out results/

Reports **achieved memory bandwidth as a fraction of device peak**, not TFLOPs.
These operators are memory bound, so GB/s is the meaningful number and TFLOPs
is noise. A good fused elementwise kernel reaches 75-90% of peak; if it reaches
20%, the problem is occupancy or masking rather than arithmetic.

Timing correctness matters as much as the kernel here:

* `triton.testing.do_bench` flushes the L2 cache between iterations. Without
  that, a 16 MB tensor stays resident in a 50 MB L2 and you measure cache
  bandwidth, which on an H100 is several times HBM bandwidth. This is the
  single most common way kernel benchmarks produce impossible numbers.
* CUDA is asynchronous, so any timing loop without a synchronise measures
  launch overhead. `do_bench` handles this; hand-rolled `time.time()` loops
  usually do not.
* The baseline must include `torch.compile`, not just eager. A kernel that
  beats eager and loses to the compiler is not a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
    import triton
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"This benchmark requires torch and triton on a CUDA machine ({e}).\n"
        "On a machine without a GPU, run ./reference.py instead: it validates "
        "the algorithms and predicts the speedup from the memory traffic model."
    )

from kernels import (
    fused_add_rmsnorm,
    fused_rmsnorm,
    fused_softmax,
    online_softmax,
    torch_rmsnorm_naive,
    torch_softmax,
)


def device_peak_bw_gb_s() -> float:
    """Peak HBM bandwidth for the current device, from its own properties."""
    p = torch.cuda.get_device_properties(0)
    # memory_clock_rate is in kHz, bus width in bits, DDR so x2.
    clock_khz = getattr(p, "memory_clock_rate", 0)
    bus_bits = getattr(p, "memory_bus_width", 0)
    if clock_khz and bus_bits:
        return clock_khz * 1e3 * bus_bits * 2 / 8 / 1e9
    known = {"H100": 3350.0, "H200": 4800.0, "A100": 2039.0,
             "L40S": 864.0, "L4": 300.0, "A10G": 600.0, "B200": 8000.0}
    for k, v in known.items():
        if k.lower() in p.name.lower():
            return v
    return float("nan")


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def bench_rmsnorm(args) -> list[dict]:
    rows: list[dict] = []
    peak = device_peak_bw_gb_s()
    dtype = getattr(torch, args.dtype)
    print(f"RMSNorm  hidden={args.hidden}  dtype={args.dtype}  "
          f"device={torch.cuda.get_device_name(0)}  peak={peak:.0f} GB/s\n")
    hdr = (f"{'tokens':>9}{'torch ms':>10}{'compile ms':>12}{'triton ms':>11}"
           f"{'speedup':>9}{'GB/s':>9}{'% peak':>8}")
    print(hdr); print("-" * len(hdr))
    for n in args.tokens:
        x = torch.randn(n, args.hidden, device="cuda", dtype=dtype)
        w = torch.randn(args.hidden, device="cuda", dtype=dtype)

        out_ref = torch_rmsnorm_naive(x, w)
        out_tri = fused_rmsnorm(x, w)
        max_err = (out_ref.float() - out_tri.float()).abs().max().item()
        tol = 2e-2 if dtype == torch.float16 else 5e-2
        if max_err > tol:
            print(f"  CORRECTNESS FAILURE at n={n}: max err {max_err:.4f}")
            continue

        # Tensors are bound as defaults so each lambda closes over THIS
        # iteration's tensors rather than reading the loop variable at call
        # time. do_bench happens to call them immediately, so the behaviour is
        # the same either way, but a timing harness whose correctness depends
        # on call ordering is one refactor away from benchmarking the wrong
        # shape and not noticing.
        ms_torch = triton.testing.do_bench(lambda x=x, w=w: torch_rmsnorm_naive(x, w))
        ms_tri = triton.testing.do_bench(lambda x=x, w=w: fused_rmsnorm(x, w))
        ms_comp = float("nan")
        if not args.skip_compile:
            try:
                compiled = torch.compile(torch_rmsnorm_naive, dynamic=False)
                compiled(x, w)  # warm the compile
                ms_comp = triton.testing.do_bench(
                    lambda f=compiled, x=x, w=w: f(x, w))
            except Exception as e:
                print(f"  torch.compile failed ({type(e).__name__}); skipping")
                args.skip_compile = True

        # fused traffic: read x, read w (negligible), write out
        nbytes = 2 * n * args.hidden * x.element_size()
        bw = gbps(nbytes, ms_tri)
        print(f"{n:>9}{ms_torch:>10.3f}{ms_comp:>12.3f}{ms_tri:>11.3f}"
              f"{ms_torch/ms_tri:>8.2f}x{bw:>9.0f}{bw/peak*100:>7.1f}%")
        rows.append({"op": "rmsnorm", "tokens": n, "hidden": args.hidden,
                     "ms_torch": ms_torch, "ms_compile": ms_comp, "ms_triton": ms_tri,
                     "speedup_vs_eager": ms_torch / ms_tri,
                     "speedup_vs_compile": ms_comp / ms_tri if ms_comp == ms_comp else None,
                     "gb_s": bw, "pct_peak": bw / peak * 100, "max_err": max_err})
    return rows


def bench_add_rmsnorm(args) -> list[dict]:
    rows: list[dict] = []
    peak = device_peak_bw_gb_s()
    dtype = getattr(torch, args.dtype)
    print(f"\nAdd+RMSNorm (the shape that appears in a transformer block)  "
          f"hidden={args.hidden}\n")
    hdr = f"{'tokens':>9}{'torch ms':>10}{'triton ms':>11}{'speedup':>9}{'GB/s':>9}{'% peak':>8}"
    print(hdr); print("-" * len(hdr))

    def torch_add_rmsnorm(x, r, w):
        r2 = x + r
        return torch_rmsnorm_naive(r2, w), r2

    for n in args.tokens:
        x = torch.randn(n, args.hidden, device="cuda", dtype=dtype)
        r = torch.randn(n, args.hidden, device="cuda", dtype=dtype)
        w = torch.randn(args.hidden, device="cuda", dtype=dtype)
        o_ref, r_ref = torch_add_rmsnorm(x, r, w)
        o_tri, r_tri = fused_add_rmsnorm(x, r, w)
        err = (o_ref.float() - o_tri.float()).abs().max().item()
        if err > (2e-2 if dtype == torch.float16 else 5e-2):
            print(f"  CORRECTNESS FAILURE at n={n}: max err {err:.4f}")
            continue
        ms_t = triton.testing.do_bench(
            lambda x=x, r=r, w=w: torch_add_rmsnorm(x, r, w))
        ms_k = triton.testing.do_bench(
            lambda x=x, r=r, w=w: fused_add_rmsnorm(x, r, w))
        nbytes = 4 * n * args.hidden * x.element_size()
        bw = gbps(nbytes, ms_k)
        print(f"{n:>9}{ms_t:>10.3f}{ms_k:>11.3f}{ms_t/ms_k:>8.2f}x"
              f"{bw:>9.0f}{bw/peak*100:>7.1f}%")
        rows.append({"op": "add_rmsnorm", "tokens": n, "hidden": args.hidden,
                     "ms_torch": ms_t, "ms_triton": ms_k,
                     "speedup_vs_eager": ms_t / ms_k, "gb_s": bw,
                     "pct_peak": bw / peak * 100})
    return rows


def bench_softmax(args) -> list[dict]:
    rows: list[dict] = []
    peak = device_peak_bw_gb_s()
    dtype = getattr(torch, args.dtype)
    print(f"\nSoftmax  cols={args.cols}  dtype={args.dtype}\n")
    hdr = (f"{'rows':>9}{'torch ms':>10}{'triton ms':>11}{'online ms':>11}"
           f"{'speedup':>9}{'GB/s':>9}{'% peak':>8}")
    print(hdr); print("-" * len(hdr))
    for n in args.tokens:
        x = torch.randn(n, args.cols, device="cuda", dtype=dtype)
        ref = torch_softmax(x)
        tri = fused_softmax(x)
        err = (ref.float() - tri.float()).abs().max().item()
        if err > 1e-2:
            print(f"  CORRECTNESS FAILURE at n={n}: max err {err:.4f}")
            continue
        on = online_softmax(x)
        err_on = (ref.float() - on.float()).abs().max().item()
        ms_t = triton.testing.do_bench(lambda x=x: torch_softmax(x))
        ms_k = triton.testing.do_bench(lambda x=x: fused_softmax(x))
        ms_o = triton.testing.do_bench(lambda x=x: online_softmax(x))
        nbytes = 2 * n * args.cols * x.element_size()
        bw = gbps(nbytes, ms_k)
        print(f"{n:>9}{ms_t:>10.3f}{ms_k:>11.3f}{ms_o:>11.3f}"
              f"{ms_t/ms_k:>8.2f}x{bw:>9.0f}{bw/peak*100:>7.1f}%")
        rows.append({"op": "softmax", "rows": n, "cols": args.cols,
                     "ms_torch": ms_t, "ms_triton": ms_k, "ms_online": ms_o,
                     "speedup_vs_eager": ms_t / ms_k, "gb_s": bw,
                     "pct_peak": bw / peak * 100,
                     "max_err": err, "max_err_online": err_on})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", choices=["rmsnorm", "softmax", "add_rmsnorm", "all"])
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device. Run ./reference.py instead.")

    rows: list[dict] = []
    if args.op in ("rmsnorm", "all"):
        rows += bench_rmsnorm(args)
    if args.op in ("add_rmsnorm", "all"):
        rows += bench_add_rmsnorm(args)
    if args.op in ("softmax", "all"):
        rows += bench_softmax(args)

    print("\nInterpretation:")
    print("  The predicted speedup from the memory-traffic model (./reference.py")
    print("  traffic) is ~2.5x for RMSNorm and ~3x for softmax against unfused")
    print("  eager PyTorch. A measured speedup far above that means the baseline")
    print("  was broken; far below means the kernel is not saturating bandwidth.")
    print("  torch.compile fuses these too, and beating IT is the real bar.")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        p = out / f"kernel-bench-{args.op}.json"
        p.write_text(json.dumps({
            "device": torch.cuda.get_device_name(0),
            "peak_bw_gb_s": device_peak_bw_gb_s(),
            "torch": torch.__version__, "triton": triton.__version__,
            "dtype": args.dtype, "rows": rows,
        }, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
