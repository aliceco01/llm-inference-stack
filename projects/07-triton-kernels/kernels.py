"""Fused Triton kernels: softmax, RMSNorm, and RMSNorm+residual.

These are real kernels and they require CUDA. `reference.py` implements the
same algorithms in NumPy so the numerics can be validated (and the algorithms
understood) on any machine.

Why fusion is the whole game here
---------------------------------
RMSNorm and softmax are **memory bound**. Their arithmetic intensity is a
handful of FLOPs per byte, far below the ~150 FLOP/byte ridge point of an H100,
so runtime is set by HBM traffic and nothing else. The optimisation is
therefore not "do less math", it is "touch memory fewer times".

A naive PyTorch RMSNorm executes as separate kernels:

    x2   = x * x          # read x, write x2
    ms   = x2.mean(-1)    # read x2, write ms
    rstd = rsqrt(ms+eps)  # read ms, write rstd
    out  = x * rstd * w   # read x, read rstd, read w, write out

That is roughly 4 reads and 4 writes of an NxD tensor. The fused kernel does
1 read and 1 write, keeping the intermediate in registers/SRAM. The expected
speedup is the traffic ratio, about 3-4x, and measuring anything far above that
means the baseline was not what you thought it was.

This is also why "achieved bandwidth" is the correct metric for these kernels,
not TFLOPs. `bench.py` reports GB/s against the device peak; a good fused
elementwise kernel reaches 75-90% of peak, and if yours reaches 20% the problem
is occupancy or masking, not arithmetic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused softmax
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. The entire row is held in SRAM.

    The max subtraction is not optional. exp(x) overflows fp16 above ~11 and
    fp32 above ~88, and attention logits routinely exceed both. Subtracting the
    row max makes every exponent <= 0 without changing the result, since
    softmax is invariant to a constant shift.
    """
    row_idx = tl.program_id(0)
    in_row = in_ptr + row_idx * in_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # -inf for masked lanes so they contribute nothing to the max or the sum.
    x = tl.load(in_row + cols, mask=mask, other=-float("inf"))
    # Reduce in fp32 regardless of input dtype: an fp16 sum over 4096 terms
    # loses several bits, and this is exactly where quantized attention
    # implementations go subtly wrong.
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)

    out_row = out_ptr + row_idx * out_row_stride
    tl.store(out_row + cols, out, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dimension."""
    assert x.is_cuda, "Triton kernels require CUDA"
    x = x.contiguous()
    n_rows, n_cols = x.shape
    # Rows must fit in SRAM for the single-pass formulation.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        raise ValueError(
            f"row of {n_cols} exceeds the single-pass limit. Use an online "
            "(streaming) softmax, which is what FlashAttention does."
        )
    # Wider rows need more warps to keep the SM busy during the reduction.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Online (streaming) softmax: the FlashAttention trick
# ---------------------------------------------------------------------------
@triton.jit
def _online_softmax_kernel(
    out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax for rows too wide to hold in SRAM.

    Single pass over the data maintaining a running max `m` and running sum
    `d`. When a new block raises the max, the accumulated sum is rescaled by
    exp(m_old - m_new) rather than recomputed. This is the identity that makes
    FlashAttention possible: it lets you compute an exact softmax while only
    ever holding a tile in fast memory.

    Note this kernel makes two passes over HBM (one to build m/d, one to write
    the normalised output) because the output needs the final denominator.
    FlashAttention avoids the second read by fusing the PV matmul into the same
    pass and rescaling the accumulator instead.
    """
    row_idx = tl.program_id(0)
    in_row = in_ptr + row_idx * in_row_stride

    m = -float("inf")
    d = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # rescale the old accumulator to the new max, then add this block
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    out_row = out_ptr + row_idx * out_row_stride
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tl.store(out_row + cols, tl.exp(x - m) / d, mask=mask)


def online_softmax(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda, "Triton kernels require CUDA"
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _online_softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols,
        BLOCK_SIZE=block_size, num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Fused RMSNorm
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
    ],
    key=["n_cols"],
)
@triton.jit
def _rmsnorm_kernel(
    out_ptr, in_ptr, w_ptr,
    in_row_stride, out_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm: out = x / sqrt(mean(x^2) + eps) * w

    No mean subtraction, unlike LayerNorm. That is the entire difference, and
    it is why RMSNorm is cheaper: one reduction instead of two, and no need to
    keep the mean around for the backward pass.

    Two loops over the row, but the second read almost always hits L2 since the
    row was just touched, so in practice this behaves close to single-pass. The
    accumulator is fp32 even for fp16/bf16 input: summing 8192 squared bf16
    values in bf16 loses enough precision to shift the norm visibly.
    """
    row_idx = tl.program_id(0)
    in_row = in_ptr + row_idx * in_row_stride
    out_row = out_ptr + row_idx * out_row_stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row + cols, (x * rstd * w).to(out_ptr.dtype.element_ty), mask=mask)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda, "Triton kernels require CUDA"
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x2)
    _rmsnorm_kernel[(x2.shape[0],)](
        out, x2, weight, x2.stride(0), out.stride(0), x2.shape[1], eps,
    )
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Fused residual add + RMSNorm
# ---------------------------------------------------------------------------
@triton.jit
def _add_rmsnorm_kernel(
    out_ptr, resid_out_ptr, in_ptr, resid_ptr, w_ptr,
    row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """residual = x + residual; out = rmsnorm(residual) * w

    This is the shape that actually appears in a transformer block, and fusing
    it saves a full read and write of the hidden state per layer. On a 70B model
    at 80 layers that is 160 avoided round trips through HBM per token, which is
    not a rounding error on a bandwidth-bound workload.

    The updated residual is also written out, because the next block needs it.
    """
    row_idx = tl.program_id(0)
    base = row_idx * row_stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(resid_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(resid_out_ptr + base + cols,
                 s.to(resid_out_ptr.dtype.element_ty), mask=mask)
        acc += s * s
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        s = tl.load(resid_out_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + cols,
                 (s * rstd * w).to(out_ptr.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float = 1e-6
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda, "Triton kernels require CUDA"
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    r2 = residual.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x2)
    resid_out = torch.empty_like(r2)
    n_cols = x2.shape[1]
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4096)
    _add_rmsnorm_kernel[(x2.shape[0],)](
        out, resid_out, x2, r2, weight, x2.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
    )
    return out.reshape(shape), resid_out.reshape(shape)


# ---------------------------------------------------------------------------
# PyTorch baselines, written the way people actually write them
# ---------------------------------------------------------------------------
def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, axis=-1)


def torch_rmsnorm_naive(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The unfused version, as it appears in a reference implementation.

    Each line is a separate CUDA kernel with its own HBM round trip.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x.to(dtype) * w)


def torch_rmsnorm_compiled(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """torch.compile fuses this too, and is the honest baseline to beat.

    A Triton kernel that beats eager PyTorch but loses to torch.compile is not
    a result worth shipping, and comparing only against eager is the most
    common way kernel speedups get overstated.
    """
    return torch.compile(torch_rmsnorm_naive, dynamic=False)(x, w, eps)
