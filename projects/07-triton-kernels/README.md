# 07 - Triton Kernel from Scratch

> Kernel intuition separates infra engineers from API callers.

Fused Triton kernels for softmax, RMSNorm, and residual-add + RMSNorm,
benchmarked against PyTorch eager **and** `torch.compile`.

```bash
# no GPU: validate the algorithms and predict the speedup
./reference.py verify
./reference.py traffic

# CUDA: benchmark
./bench.py all --hidden 4096 --out results/
pytest test_kernels.py -v
```

`kernels.py` and `bench.py` require CUDA. `reference.py` and the NumPy half of
the test suite run anywhere, and they cover the two things that are actually
subtle: the online-softmax rescaling identity and the fp32 accumulation
requirement. If the algorithm is wrong there, the Triton kernel is wrong too,
and no amount of occupancy tuning will save it.

## These operators are memory bound, and that determines everything

RMSNorm and softmax do a handful of FLOPs per byte, far below the ~150
FLOP/byte ridge point of an H100. Runtime is set by HBM traffic and nothing
else. So the optimisation is not "do less math", it is "touch memory fewer
times", and the achievable speedup is knowable in advance as a traffic ratio.

Unfused RMSNorm in PyTorch is four separate CUDA kernels:

```
x2   = x * x           # read x,  write x2
ms   = x2.mean(-1)     # read x2, write ms
rstd = rsqrt(ms + eps) # read ms, write rstd
out  = x * rstd * w    # read x, read rstd, write out
```

`./reference.py traffic` counts the bytes. For a 4096 x 4096 bf16 tensor, one
full pass over the tensor is `t` = 33.55 MB, and the totals are:

```
rmsnorm        unfused  5t = 167.8 MB   fused  2t =  67.1 MB   predicted speedup  2.50x
softmax        unfused  6t = 201.3 MB   fused  2t =  67.1 MB   predicted speedup  3.00x
add_rmsnorm    unfused  8t = 268.4 MB   fused  4t = 134.2 MB   predicted speedup  2.00x
```

(Per-row scalar tensors add a further 4 x 8 KB, which is negligible. The full
per-step breakdown is in the command's output.)

**That ratio is the number to measure against**, and it cuts both ways:

- A measured speedup far *above* the ratio means the baseline was broken, not
  that the kernel is brilliant. Look for stray dtype casts, non-contiguous
  input, or a missing synchronise in the timing loop.
- A measured speedup far *below* it means the kernel is not saturating
  bandwidth. Check occupancy, block size, and whether masking is forcing
  partial cache-line transactions.

Report **GB/s against device peak**, not TFLOPs. For these operators TFLOPs is
a meaningless number. A good fused elementwise kernel reaches 75-90% of peak.

## The three kernels

**`fused_softmax`** holds an entire row in SRAM, one program per row. Subtracts
the row max before exponentiating, which is not optional: `exp` overflows fp16
above ~11 and fp32 above ~88, and attention logits routinely exceed both.
Softmax is invariant to a constant shift, so this is free correctness.

**`online_softmax`** is the streaming version for rows too wide for SRAM, and
it is the identity behind FlashAttention. Maintaining a running max `m` and sum
`d`, when a new block raises the max you rescale rather than recompute:

```
d_new = d_old * exp(m_old - m_new) + sum(exp(x_block - m_new))
```

This is **exact, not an approximation**, which `test_online_softmax_is_exact`
verifies to 1e-12 across block sizes, including the worst case where the row
maximum arrives in the final block. It is what lets attention compute an exact
softmax without ever materialising the N x N score matrix. Note this kernel
still makes two HBM passes, because the output needs the final denominator;
FlashAttention avoids the second read by fusing the PV matmul into the same
pass and rescaling its accumulator instead.

**`fused_add_rmsnorm`** is the shape that actually appears in a transformer
block. Fusing the residual add saves a full read and write of the hidden state
per layer: on a 70B model at 80 layers that is 160 avoided HBM round trips per
token.

### Why the accumulator is fp32

Every reduction accumulates in fp32 regardless of input dtype. Summing 8192
squared bf16 values in bf16 loses enough precision to shift the norm visibly.
`test_fp32_accumulation_beats_fp16` asserts the fp16 accumulator is more than
10x worse than fp32 against an fp64 reference. This is exactly where
low-precision attention implementations go subtly wrong: the kernel looks
correct, the outputs are plausible, and the model is quietly degraded.

## Benchmark methodology

Timing these kernels correctly is as easy to get wrong as writing them.

- **L2 cache flushing.** `triton.testing.do_bench` flushes L2 between
  iterations. Without it, a 16 MB tensor stays resident in a 50 MB L2 and you
  measure cache bandwidth, which on an H100 is several times HBM bandwidth.
  This is the most common source of impossible kernel numbers.
- **CUDA is asynchronous.** A hand-rolled `time.time()` loop without a
  synchronise measures kernel launch overhead, not kernel time.
- **`torch.compile` is the real baseline.** It fuses these patterns too. A
  kernel that beats eager and loses to the compiler is not a result worth
  shipping, and comparing only against eager is how kernel speedups get
  overstated. `bench.py` reports both columns.
- **Correctness is checked before timing.** Every benchmark shape validates
  against the PyTorch reference first and refuses to report a time for a shape
  that fails, because a fast wrong kernel is worthless.

## What the test suite covers

Non-power-of-two widths (1, 3, 17, 129, 1023, 4095) catch masking bugs where
tail lanes contaminate the reduction, which is the single most common Triton
error. Extreme-magnitude inputs confirm the max subtraction works. RMSNorm
scale invariance and the RMSNorm-is-not-LayerNorm test catch the two ways the
normalisation itself gets implemented wrong.

## Autotuning

`_rmsnorm_kernel` carries a `triton.autotune` decorator over block size and
warp count keyed on `n_cols`. Occupancy for these kernels depends on the
hardware's SRAM per SM and the row width, and the best configuration is not
predictable from first principles: 4096 columns wants different warps on an
A100 than on an H100. Autotuning costs a one-time search per shape and removes
an entire category of hand-tuning that goes stale on the next GPU generation.
