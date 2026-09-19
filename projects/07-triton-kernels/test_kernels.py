"""Correctness tests.

The NumPy algorithm tests run anywhere. The Triton tests are skipped without
CUDA, which is the right behaviour: a skipped GPU test is honest, a GPU test
that silently passes on CPU is not.

    pytest test_kernels.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import reference as ref

# --------------------------------------------------------------------------
# Algorithm tests: no GPU required
# --------------------------------------------------------------------------


def test_stable_matches_naive_in_safe_range():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, size=(8, 512))
    assert np.abs(ref.softmax_naive(x) - ref.softmax_stable(x)).max() < 1e-12


def test_naive_overflows_where_stable_does_not():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, size=(4, 256)) + 800.0
    with np.errstate(over="ignore", invalid="ignore"):
        naive = ref.softmax_naive(x)
    stable = ref.softmax_stable(x)
    assert not np.isfinite(naive).all(), "expected naive softmax to overflow"
    assert np.isfinite(stable).all()
    np.testing.assert_allclose(stable.sum(axis=-1), 1.0, atol=1e-12)


@pytest.mark.parametrize("block", [8, 16, 128, 1024])
def test_online_softmax_is_exact(block):
    """The rescaling identity must be exact for any block size."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 5, size=(16, 2048))
    assert np.abs(ref.softmax_online(x, block=block) - ref.softmax_stable(x)).max() < 1e-12


def test_online_softmax_late_max():
    """Worst case: the row maximum arrives in the final block."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(4, 1024))
    x[:, -1] = 500.0
    assert np.abs(ref.softmax_online(x, block=64) - ref.softmax_stable(x)).max() < 1e-12


def test_online_softmax_all_equal():
    x = np.full((3, 512), -7.5)
    out = ref.softmax_online(x, block=32)
    np.testing.assert_allclose(out, 1.0 / 512, rtol=1e-12)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 10, size=(32, 777))
    np.testing.assert_allclose(ref.softmax_stable(x).sum(axis=-1), 1.0, atol=1e-12)


def test_fp32_accumulation_beats_fp16():
    """Justifies the fp32 accumulator in the kernel."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, size=(4, 8192)).astype(np.float32)
    w = np.ones(8192, dtype=np.float32)
    exact = ref.rmsnorm(x, w, accum_dtype=np.float64)
    e32 = np.abs(ref.rmsnorm(x, w, accum_dtype=np.float32) - exact).max()
    x16, w16 = x.astype(np.float16), w.astype(np.float16)
    e16 = np.abs(ref.rmsnorm(x16, w16, accum_dtype=np.float16).astype(np.float32) - exact).max()
    assert e16 > e32 * 10, f"expected fp16 accumulation to be much worse: {e16} vs {e32}"


def test_rmsnorm_is_not_layernorm():
    rng = np.random.default_rng(6)
    x = rng.normal(5.0, 1.0, size=(4, 256))   # non-zero mean
    w = np.ones(256)
    assert np.abs(ref.rmsnorm(x, w) - ref.layernorm(x, w)).max() > 0.1


def test_rmsnorm_scale_invariance():
    """RMSNorm(c*x) == RMSNorm(x) for c > 0, up to eps."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=(4, 512))
    w = np.ones(512)
    a = ref.rmsnorm(x, w, eps=1e-12, accum_dtype=np.float64)
    b = ref.rmsnorm(x * 7.0, w, eps=1e-12, accum_dtype=np.float64)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_traffic_model_predicts_fusion_gain():
    m = ref.traffic_model(4096, 4096)
    for op in ("rmsnorm", "softmax", "add_rmsnorm"):
        uf = m[op]["unfused"]["_total"]
        f = m[op]["fused"]["_total"]
        assert uf > f, f"{op}: fusion should reduce traffic"
        assert 1.5 <= uf / f <= 4.0, f"{op}: implausible traffic ratio {uf/f}"


# --------------------------------------------------------------------------
# Triton tests: require CUDA
# --------------------------------------------------------------------------
cuda = pytest.importorskip("torch", reason="torch not installed")
HAS_CUDA = cuda.cuda.is_available() if hasattr(cuda, "cuda") else False
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")


@requires_cuda
@pytest.mark.parametrize("n_cols", [128, 512, 1000, 4096])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_triton_softmax_matches_torch(n_cols, dtype_name):
    import torch
    from kernels import fused_softmax
    dtype = getattr(torch, dtype_name)
    x = torch.randn(64, n_cols, device="cuda", dtype=dtype)
    got = fused_softmax(x).float()
    want = torch.softmax(x.float(), dim=-1)
    assert (got - want).abs().max().item() < 1e-2


@requires_cuda
def test_triton_softmax_non_power_of_two():
    """Masking correctness: the tail lanes must not contaminate the reduction."""
    import torch
    from kernels import fused_softmax
    for n_cols in (1, 3, 17, 129, 1023, 4095):
        x = torch.randn(8, n_cols, device="cuda", dtype=torch.float32)
        got = fused_softmax(x)
        want = torch.softmax(x, dim=-1)
        assert (got - want).abs().max().item() < 1e-5, f"failed at n_cols={n_cols}"
        torch.testing.assert_close(got.sum(-1), torch.ones(8, device="cuda"), atol=1e-5, rtol=1e-5)


@requires_cuda
def test_triton_softmax_extreme_values():
    import torch
    from kernels import fused_softmax
    x = torch.randn(8, 1024, device="cuda", dtype=torch.float32) + 1000.0
    got = fused_softmax(x)
    assert torch.isfinite(got).all(), "kernel must not overflow on large logits"
    torch.testing.assert_close(got.sum(-1), torch.ones(8, device="cuda"), atol=1e-5, rtol=1e-5)


@requires_cuda
@pytest.mark.parametrize("hidden", [512, 4096, 8192, 11008])
def test_triton_rmsnorm_matches_torch(hidden):
    import torch
    from kernels import fused_rmsnorm, torch_rmsnorm_naive
    x = torch.randn(128, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    got = fused_rmsnorm(x, w).float()
    want = torch_rmsnorm_naive(x, w).float()
    assert (got - want).abs().max().item() < 5e-2


@requires_cuda
def test_triton_online_softmax_matches():
    import torch
    from kernels import online_softmax
    x = torch.randn(32, 8192, device="cuda", dtype=torch.float32)
    got = online_softmax(x, block_size=1024)
    want = torch.softmax(x, dim=-1)
    assert (got - want).abs().max().item() < 1e-5


@requires_cuda
def test_triton_add_rmsnorm_updates_residual():
    import torch
    from kernels import fused_add_rmsnorm, torch_rmsnorm_naive
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out, resid = fused_add_rmsnorm(x, r, w)
    torch.testing.assert_close(resid.float(), (x + r).float(), atol=1e-2, rtol=1e-2)
    want = torch_rmsnorm_naive(x + r, w).float()
    assert (out.float() - want).abs().max().item() < 5e-2
