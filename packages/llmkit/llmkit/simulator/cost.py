"""Roofline cost model for a single engine step.

The model is deliberately simple because the physics is simple, and getting
the two regimes right explains almost every latency curve in LLM serving:

PREFILL is compute bound. Work scales with (tokens x parameters), so a step
processing T tokens costs roughly 2 * P_active * T FLOPs plus an attention term
that grows with T^2. Doubling the prompt doubles the cost, and doubling it
again does more than that once attention dominates.

DECODE is memory-bandwidth bound. Every step reads the entire weight matrix
from HBM to generate ONE token per sequence, so the weight traffic is fixed
regardless of batch size. That is the whole reason continuous batching works:
going from batch 1 to batch 64 costs almost nothing extra in weight traffic and
multiplies throughput by ~64, until KV cache reads (which DO scale with batch
and context) take over as the dominant term.

The crossover between those two is where every interesting scheduling decision
lives, and it is why long-context requests starve decode without chunked
prefill.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..gpus import GPUSpec
from ..modelspec import DTYPE_BYTES, DType, ModelSpec


@dataclass
class EfficiencyModel:
    """Fraction of peak actually achieved.

    Peak numbers are marketing. These defaults reflect what well-tuned kernels
    reach in practice: FlashAttention-class prefill lands near 60-70% of peak
    FLOPs, and decode GEMV reaches 75-85% of peak bandwidth.
    """

    prefill_flops_eff: float = 0.65
    decode_bw_eff: float = 0.80
    attn_flops_eff: float = 0.50     # attention kernels are less efficient than GEMM
    step_overhead_ms: float = 1.2    # scheduler + sampling + kernel launch
    cuda_graph_overhead_ms: float = 0.35   # decode overhead with graphs captured
    tp_allreduce_eff: float = 0.75   # fraction of NVLink bandwidth realised


@dataclass
class StepCost:
    duration_ms: float
    prefill_tokens: int = 0
    decode_seqs: int = 0
    compute_ms: float = 0.0
    memory_ms: float = 0.0
    attn_ms: float = 0.0
    comm_ms: float = 0.0
    overhead_ms: float = 0.0
    bound_by: str = "overhead"


class CostModel:
    def __init__(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        *,
        tp: int = 1,
        pp: int = 1,
        weight_dtype: DType = "bf16",
        kv_dtype: DType = "fp16",
        eff: EfficiencyModel | None = None,
        cuda_graphs: bool = True,
    ) -> None:
        self.m = model
        self.g = gpu
        self.tp = tp
        self.pp = pp
        self.weight_dtype = weight_dtype
        self.kv_dtype = kv_dtype
        self.eff = eff or EfficiencyModel()
        self.cuda_graphs = cuda_graphs

        world = tp * pp
        self.weight_bytes = model.weight_bytes(weight_dtype) / world
        self.active_params = model.active_params() / world
        self.kv_bytes_per_token = model.kv_bytes_per_token(kv_dtype) / max(tp, 1)
        self.peak_flops = gpu.tflops(weight_dtype) * 1e12
        self.bw = gpu.bandwidth_gb_s * 1e9

    # ------------------------------------------------------------------
    def prefill_ms(self, chunk_lens: Sequence[int], ctx_lens: Sequence[int] | None = None) -> StepCost:
        """Cost of prefilling the given chunks.

        `ctx_lens` is the context each chunk attends over, which differs from
        the chunk length under chunked prefill: chunk 3 of a long prompt is
        short but attends over everything before it, so its attention cost is
        not small. Models that ignore this predict chunked prefill is free.
        """
        T = sum(chunk_lens)
        if T == 0:
            return StepCost(0.0)
        ctx_lens = list(ctx_lens or chunk_lens)

        # Dense/GEMM term: 2 FLOPs per parameter per token.
        gemm_flops = 2.0 * self.active_params * T
        gemm_ms = gemm_flops / (self.peak_flops * self.eff.prefill_flops_eff) * 1e3

        # Attention term. For a chunk of length c attending over context n,
        # the score matrix is c x n, and QK^T plus PV are 2 * 2 * c * n * d
        # FLOPs per head per layer (halved for the causal mask when c == n).
        attn_flops = 0.0
        d = self.m.head_dim
        h = self.m.num_attention_heads / max(self.tp, 1)
        L = self.m.num_layers / max(self.pp, 1)
        # strict=True: a chunk without a matching context length would be
        # silently dropped, under-counting attention cost in exactly the
        # chunked-prefill case this function exists to model correctly.
        for c, n in zip(chunk_lens, ctx_lens, strict=True):
            causal = 0.5 if c == n else 1.0
            attn_flops += 2 * 2 * c * n * d * h * L * causal
        attn_ms = attn_flops / (self.peak_flops * self.eff.attn_flops_eff) * 1e3

        # Weights still have to be read, which matters for small chunks.
        mem_ms = self.weight_bytes / (self.bw * self.eff.decode_bw_eff) * 1e3

        comm_ms = self._allreduce_ms(T)
        compute_ms = gemm_ms + attn_ms
        body = max(compute_ms, mem_ms)
        overhead = self.eff.step_overhead_ms
        total = body + comm_ms + overhead
        return StepCost(
            duration_ms=total, prefill_tokens=T, compute_ms=gemm_ms,
            memory_ms=mem_ms, attn_ms=attn_ms, comm_ms=comm_ms,
            overhead_ms=overhead,
            bound_by="compute" if compute_ms >= mem_ms else "memory",
        )

    def decode_ms(self, ctx_lens: Sequence[int]) -> StepCost:
        """Cost of one decode step over a batch of sequences.

        Weight traffic is constant in batch size; KV traffic scales with the
        sum of context lengths. The ratio of those two terms is the single
        number that decides whether batching still helps.
        """
        B = len(ctx_lens)
        if B == 0:
            return StepCost(0.0)
        total_ctx = sum(ctx_lens)

        kv_bytes = total_ctx * self.kv_bytes_per_token
        mem_bytes = self.weight_bytes + kv_bytes
        mem_ms = mem_bytes / (self.bw * self.eff.decode_bw_eff) * 1e3

        gemm_flops = 2.0 * self.active_params * B
        compute_ms = gemm_flops / (self.peak_flops * self.eff.prefill_flops_eff) * 1e3

        comm_ms = self._allreduce_ms(B)
        overhead = (self.eff.cuda_graph_overhead_ms if self.cuda_graphs
                    else self.eff.step_overhead_ms)
        body = max(mem_ms, compute_ms)
        total = body + comm_ms + overhead
        return StepCost(
            duration_ms=total, decode_seqs=B, compute_ms=compute_ms,
            memory_ms=mem_ms, comm_ms=comm_ms, overhead_ms=overhead,
            bound_by="memory" if mem_ms >= compute_ms else "compute",
        )

    def mixed_ms(self, chunk_lens: Sequence[int], prefill_ctx: Sequence[int],
                 decode_ctx: Sequence[int]) -> StepCost:
        """Chunked prefill: prefill chunks and decodes share one batch.

        The costs are additive in the compute/memory terms but the fixed
        per-step overhead is paid once, which is precisely the efficiency
        chunked prefill buys.
        """
        p = self.prefill_ms(chunk_lens, prefill_ctx)
        d = self.decode_ms(decode_ctx)
        if p.prefill_tokens == 0:
            return d
        if d.decode_seqs == 0:
            return p
        compute = p.compute_ms + p.attn_ms + d.compute_ms
        # Weights are read once for the fused batch, not once per sub-batch.
        memory = max(p.memory_ms, d.memory_ms) + (
            d.memory_ms - self.weight_bytes / (self.bw * self.eff.decode_bw_eff) * 1e3
        )
        comm = max(p.comm_ms, d.comm_ms)
        overhead = self.eff.step_overhead_ms
        body = max(compute, memory)
        return StepCost(
            duration_ms=body + comm + overhead,
            prefill_tokens=p.prefill_tokens, decode_seqs=d.decode_seqs,
            compute_ms=compute, memory_ms=memory, attn_ms=p.attn_ms,
            comm_ms=comm, overhead_ms=overhead,
            bound_by="compute" if compute >= memory else "memory",
        )

    # ------------------------------------------------------------------
    def _allreduce_ms(self, n_tokens: int) -> float:
        """Tensor-parallel all-reduce cost: 2 per layer, ring algorithm."""
        if self.tp <= 1:
            return 0.0
        link = (self.g.nvlink_gb_s or 64.0) * 1e9   # PCIe gen5 x16 fallback
        b = DTYPE_BYTES[self.weight_dtype if self.weight_dtype in DTYPE_BYTES else "bf16"]
        bytes_per_ar = n_tokens * self.m.hidden_size * b
        n_ar = 2 * self.m.num_layers / max(self.pp, 1)
        ring = 2 * (self.tp - 1) / self.tp
        return (bytes_per_ar * ring * n_ar) / (link * self.eff.tp_allreduce_eff) * 1e3

    # ------------------------------------------------------------------
    def roofline_summary(self, batch_size: int, ctx_len: int) -> dict[str, float]:
        """Diagnostic: where the decode step's time actually goes."""
        d = self.decode_ms([ctx_len] * batch_size)
        kv_bytes = batch_size * ctx_len * self.kv_bytes_per_token
        return {
            "step_ms": round(d.duration_ms, 3),
            "tokens_per_s": round(batch_size / (d.duration_ms / 1e3), 1),
            "weight_read_gib": round(self.weight_bytes / 1024 ** 3, 3),
            "kv_read_gib": round(kv_bytes / 1024 ** 3, 3),
            "kv_share_of_traffic": round(kv_bytes / (self.weight_bytes + kv_bytes), 3),
            "arithmetic_intensity": round(
                2 * self.active_params * batch_size / max(self.weight_bytes + kv_bytes, 1), 2
            ),
            "bound_by": d.bound_by,
        }
