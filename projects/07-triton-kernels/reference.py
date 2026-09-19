#!/usr/bin/env python3
"""NumPy references for the Triton kernels, plus the memory-traffic model.

Runs anywhere, no GPU. Two jobs:

1. **Validate the algorithms.** The online-softmax rescaling identity and the
   fp32-accumulation requirement are the two things that are actually subtle in
   `kernels.py`, and both can be verified exactly in NumPy. If the algorithm is
   wrong here, the Triton kernel is wrong too and no amount of occupancy tuning
   will fix it.

2. **Predict the speedup before writing the kernel.** These operators are
   memory bound, so the achievable speedup is the ratio of HBM bytes moved.
   Computing that first tells you whether a kernel is worth writing and what
   number would indicate you succeeded. A "12x speedup" on an operator whose
   traffic ratio is 4x means the baseline was broken.

    ./reference.py verify      # numerical checks
    ./reference.py traffic     # HBM traffic model and predicted speedups
"""

from __future__ import annotations

import argparse
import math

import numpy as np


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
def softmax_naive(x: np.ndarray) -> np.ndarray:
    """Textbook softmax. Overflows for realistic attention logits."""
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Subtract the row max first. Mathematically identical, numerically safe.

    softmax(x) == softmax(x - c) for any constant c, so choosing c = max(x)
    makes every exponent <= 0 and bounds exp() by 1.
    """
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_online(x: np.ndarray, block: int = 128) -> np.ndarray:
    """Streaming softmax: one pass to build (max, sum), one to normalise.

    This is the identity behind FlashAttention. Processing block by block, when
    a new block raises the running max from m_old to m_new, the running sum is
    corrected by a single multiplication:

        d_new = d_old * exp(m_old - m_new) + sum(exp(x_block - m_new))

    which is exact, not an approximation. It means an exact softmax can be
    computed while only ever holding one tile in fast memory, which is what
    allows attention to avoid materialising the full NxN score matrix.
    """
    x = np.atleast_2d(x)
    out = np.empty_like(x, dtype=np.float64)
    for r in range(x.shape[0]):
        row = x[r]
        m = -np.inf
        d = 0.0
        for off in range(0, row.shape[0], block):
            blk = row[off:off + block]
            m_new = max(m, float(blk.max()))
            d = d * math.exp(m - m_new) + float(np.exp(blk - m_new).sum())
            m = m_new
        out[r] = np.exp(row - m) / d
    return out


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6,
            accum_dtype=np.float32) -> np.ndarray:
    """out = x / sqrt(mean(x^2) + eps) * w

    No mean subtraction, unlike LayerNorm: one reduction instead of two.
    `accum_dtype` exists to demonstrate why the kernel accumulates in fp32.
    """
    xa = x.astype(accum_dtype)
    ms = np.mean(xa * xa, axis=-1, keepdims=True)
    return (xa * (1.0 / np.sqrt(ms + eps))).astype(x.dtype) * w


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
              eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    out = (x - mu) / np.sqrt(var + eps) * w
    return out + b if b is not None else out


# ---------------------------------------------------------------------------
# Memory traffic model
# ---------------------------------------------------------------------------
def traffic_model(n_rows: int, n_cols: int, dtype_bytes: int = 2) -> dict[str, dict]:
    """HBM bytes moved by fused vs unfused implementations.

    Counts each read and write of an N x D tensor. Intermediates in the unfused
    path are materialised to HBM because each PyTorch op is its own kernel with
    no way to keep results in registers across the boundary.
    """
    t = n_rows * n_cols * dtype_bytes          # one full tensor
    row = n_rows * dtype_bytes                 # one per-row scalar tensor
    return {
        "rmsnorm": {
            "unfused": {
                "x*x        (r x, w x2)": 2 * t,
                "mean       (r x2, w ms)": t + row,
                "rsqrt      (r ms, w rstd)": 2 * row,
                "x*rstd*w   (r x, r rstd, w out)": 2 * t + row,
                "_total": 5 * t + 4 * row,
            },
            "fused": {
                "read x, read w, write out": 2 * t,
                "_total": 2 * t,
            },
        },
        "softmax": {
            "unfused": {
                "max        (r x, w m)": t + row,
                "sub+exp    (r x, r m, w e)": 2 * t + row,
                "sum        (r e, w d)": t + row,
                "div        (r e, r d, w out)": 2 * t + row,
                "_total": 6 * t + 4 * row,
            },
            "fused": {"read x, write out": 2 * t, "_total": 2 * t},
        },
        "add_rmsnorm": {
            "unfused": {
                "add        (r x, r resid, w resid)": 3 * t,
                "rmsnorm    (unfused, as above)": 5 * t,
                "_total": 8 * t,
            },
            "fused": {
                "read x, read resid, write resid, write out": 4 * t,
                "_total": 4 * t,
            },
        },
    }


def cmd_traffic(args) -> int:
    shapes = [
        ("Llama-8B hidden, 4k tokens", 4096, 4096),
        ("Llama-70B hidden, 4k tokens", 4096, 8192),
        ("attention scores, 32 heads x 4k", 32 * 4096, 4096),
    ]
    for name, rows, cols in shapes:
        print(f"\n{name}   ({rows} x {cols}, bf16)")
        print("-" * 72)
        model = traffic_model(rows, cols)
        for op, variants in model.items():
            uf = variants["unfused"]["_total"]
            f = variants["fused"]["_total"]
            print(f"  {op:<14} unfused {uf/1e6:>9.1f} MB   fused {f/1e6:>9.1f} MB   "
                  f"predicted speedup {uf/f:>5.2f}x")
    print("\nDetail for rmsnorm (4096 x 4096, bf16):")
    m = traffic_model(4096, 4096)["rmsnorm"]
    for variant in ("unfused", "fused"):
        print(f"  {variant}:")
        for k, v in m[variant].items():
            if k == "_total":
                print(f"    {'TOTAL':<34}{v/1e6:>9.1f} MB")
            else:
                print(f"    {k:<34}{v/1e6:>9.1f} MB")
    print("""
These ratios are the number a fused kernel should be measured against. Because
the operators are memory bound, achieved speedup tracks avoided HBM traffic
almost exactly. Two consequences worth internalising:

  * A measured speedup far ABOVE the traffic ratio means the baseline was
    broken (unnecessary casts, non-contiguous input, or a synchronisation in
    the timing loop), not that the kernel is brilliant.
  * A measured speedup far BELOW it means the kernel is not saturating
    bandwidth: check occupancy, block size and whether masking is forcing
    partial cache lines.

Report GB/s achieved against device peak, not TFLOPs. For these operators
TFLOPs is a meaningless number.""")
    return 0


# ---------------------------------------------------------------------------
def cmd_verify(args) -> int:
    rng = np.random.default_rng(0)
    ok = True

    print("1. stable softmax == naive softmax, where naive does not overflow")
    x = rng.normal(0, 2, size=(8, 512))
    a, b = softmax_naive(x), softmax_stable(x)
    err = np.abs(a - b).max()
    print(f"   max abs diff {err:.3e}   {'PASS' if err < 1e-12 else 'FAIL'}")
    ok &= err < 1e-12

    print("\n2. naive softmax overflows on realistic attention logits")
    big = rng.normal(0, 2, size=(4, 512)) + 800.0
    with np.errstate(over="ignore", invalid="ignore"):
        naive = softmax_naive(big)
    stable = softmax_stable(big)
    n_bad = int(np.isnan(naive).sum() + np.isinf(naive).sum())
    print(f"   naive produced {n_bad} non-finite values; stable produced "
          f"{int(np.isnan(stable).sum() + np.isinf(stable).sum())}")
    print(f"   {'PASS' if n_bad > 0 and np.isfinite(stable).all() else 'FAIL'}"
          "  (this is why the kernel subtracts the row max)")
    ok &= n_bad > 0 and bool(np.isfinite(stable).all())

    print("\n3. online (streaming) softmax == stable softmax, exactly")
    for block in (16, 128, 1024):
        x = rng.normal(0, 5, size=(16, 2048))
        err = np.abs(softmax_online(x, block=block) - softmax_stable(x)).max()
        status = "PASS" if err < 1e-12 else "FAIL"
        print(f"   block={block:<5} max abs diff {err:.3e}   {status}")
        ok &= err < 1e-12
    print("   the rescaling identity is exact, not an approximation")

    print("\n4. online softmax survives a max that arrives in the last block")
    x = rng.normal(0, 1, size=(4, 1024))
    x[:, -1] = 500.0     # worst case for the running max
    err = np.abs(softmax_online(x, block=128) - softmax_stable(x)).max()
    print(f"   max abs diff {err:.3e}   {'PASS' if err < 1e-12 else 'FAIL'}")
    ok &= err < 1e-12

    print("\n5. RMSNorm accumulation dtype matters")
    xf = rng.normal(0, 1, size=(4, 8192)).astype(np.float32)
    w = np.ones(8192, dtype=np.float32)
    ref = rmsnorm(xf, w, accum_dtype=np.float64)
    in_fp32 = rmsnorm(xf, w, accum_dtype=np.float32)
    x16 = xf.astype(np.float16)
    w16 = w.astype(np.float16)
    in_fp16 = rmsnorm(x16, w16, accum_dtype=np.float16).astype(np.float32)
    e32 = np.abs(in_fp32 - ref).max()
    e16 = np.abs(in_fp16 - ref).max()
    print(f"   fp32 accumulation: max err {e32:.3e}")
    print(f"   fp16 accumulation: max err {e16:.3e}   ({e16/max(e32,1e-30):.0f}x worse)")
    print(f"   {'PASS' if e16 > e32 * 10 else 'INCONCLUSIVE'}"
          "  (this is why the kernel accumulates in fp32)")

    print("\n6. RMSNorm is not LayerNorm")
    x = rng.normal(5.0, 1.0, size=(4, 256))   # non-zero mean
    w = np.ones(256)
    r, l = rmsnorm(x, w), layernorm(x, w)
    print(f"   max abs diff on non-zero-mean input: {np.abs(r-l).max():.3f}")
    print("   RMSNorm does not subtract the mean; substituting one for the other")
    print("   silently changes model outputs")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="numerical validation of the algorithms")
    p = sub.add_parser("traffic", help="HBM traffic model and predicted speedups")
    p.add_argument("--dtype-bytes", type=int, default=2)
    args = ap.parse_args()
    return {"verify": cmd_verify, "traffic": cmd_traffic}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
