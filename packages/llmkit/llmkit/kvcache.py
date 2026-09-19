"""KV cache memory budgeting: predict before you OOM.

The claim this module exists to support is that most production OOMs are KV
cache math errors rather than model-size errors, and the arithmetic backs it
up. On an 80 GB H100 running Llama-3.1-8B in bf16, weights take 16 GB and the
remaining ~56 GB of KV cache is exactly 3.5 full-length (128k) sequences. The
model fits trivially; the *context* is what kills you, and it does so only once
real traffic arrives with long prompts.

Everything here is auditable: `MemoryBudget.explain()` prints the whole
breakdown so a number can be argued with rather than trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .gpus import GPUSpec, get_gpu
from .modelspec import DTYPE_BYTES, DType, ModelSpec, get_model

GIB = 1024 ** 3


@dataclass
class ServingConfig:
    """The knobs that decide whether a deployment fits in memory."""

    tp: int = 1                       # tensor parallel degree
    pp: int = 1                       # pipeline parallel degree
    weight_dtype: DType = "bf16"
    kv_dtype: DType = "fp16"
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    max_model_len: int = 8192
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_cuda_graphs: bool = True
    enable_prefix_caching: bool = True
    swap_space_gb: float = 4.0        # CPU-side swap for preempted sequences

    def world_size(self) -> int:
        return self.tp * self.pp


@dataclass
class MemoryBudget:
    """Where every byte of VRAM goes, per GPU."""

    gpu: str = ""
    model: str = ""
    total_gib: float = 0.0
    usable_gib: float = 0.0          # total * gpu_memory_utilization
    weights_gib: float = 0.0
    activation_gib: float = 0.0
    cuda_graph_gib: float = 0.0
    framework_overhead_gib: float = 0.0
    kv_cache_gib: float = 0.0

    kv_bytes_per_token: float = 0.0
    block_bytes: float = 0.0
    num_blocks: int = 0
    max_cached_tokens: int = 0

    kv_heads_per_gpu: int = 0
    kv_heads_replicated: bool = False
    # Sliding-window models stop growing their cache at the window size, so
    # capacity questions must be asked about the *effective* length, not the
    # nominal one. Without this a windowed model looks far more expensive at
    # long context than it is.
    sliding_window: int | None = None
    windowed_layer_frac: float = 0.0
    fits: bool = True
    problems: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # --- capacity questions the budget can answer ------------------------
    def effective_seq_len(self, seq_len: int) -> float:
        """Tokens actually held in cache for a sequence of `seq_len`."""
        if not self.sliding_window:
            return float(seq_len)
        eff = min(seq_len, self.sliding_window)
        f = self.windowed_layer_frac
        return f * eff + (1.0 - f) * seq_len

    def max_concurrent_seqs(self, seq_len: int) -> int:
        """How many sequences of `seq_len` tokens fit simultaneously.

        Rounded up to whole blocks per sequence, because paging allocates in
        blocks: a 17-token sequence with block_size 16 occupies 2 blocks, and
        ignoring that rounding is a quiet 5-10% overestimate at short lengths.
        """
        if self.num_blocks <= 0:
            return 0
        bs = max(self.config.get("block_size", 16), 1)
        blocks_per_seq = math.ceil(self.effective_seq_len(seq_len) / bs)
        return self.num_blocks // max(blocks_per_seq, 1)

    def max_seq_len(self, concurrency: int) -> int:
        if concurrency <= 0 or self.num_blocks <= 0:
            return 0
        bs = self.config.get("block_size", 16)
        return (self.num_blocks // concurrency) * bs

    def headroom_ratio(self, seq_len: int, concurrency: int) -> float:
        """<1.0 means the target workload does not fit."""
        bs = self.config.get("block_size", 16)
        need = concurrency * math.ceil(self.effective_seq_len(seq_len) / bs)
        return self.num_blocks / max(need, 1)

    def explain(self) -> str:
        c = self.config
        lines = [
            f"KV cache budget: {self.model} on {c.get('world_size',1)}x {self.gpu} "
            f"(TP={c.get('tp')}, PP={c.get('pp')})",
            "-" * 74,
            f"  {'GPU memory (total)':<34}{self.total_gib:>10.2f} GiB",
            f"  {'x gpu_memory_utilization ' + str(c.get('gpu_memory_utilization')):<34}"
            f"{self.usable_gib:>10.2f} GiB",
            f"  {'- model weights (' + str(c.get('weight_dtype')) + ')':<34}"
            f"{-self.weights_gib:>10.2f} GiB",
            f"  {'- peak activations':<34}{-self.activation_gib:>10.2f} GiB",
            f"  {'- CUDA graphs':<34}{-self.cuda_graph_gib:>10.2f} GiB",
            f"  {'- framework/NCCL overhead':<34}{-self.framework_overhead_gib:>10.2f} GiB",
            "-" * 74,
            f"  {'= KV cache available':<34}{self.kv_cache_gib:>10.2f} GiB",
            "",
            f"  KV per token       {self.kv_bytes_per_token/1024:>10.2f} KiB/token"
            f"   (per GPU, {self.kv_heads_per_gpu} KV heads)",
            f"  Block size         {c.get('block_size'):>10} tokens"
            f"   -> {self.block_bytes/1024:.1f} KiB per block",
            f"  Blocks             {self.num_blocks:>10,}",
            f"  Cacheable tokens   {self.max_cached_tokens:>10,}",
        ]
        if self.kv_heads_replicated:
            lines.append(
                f"  NOTE: TP={c.get('tp')} exceeds {c.get('num_kv_heads')} KV heads, so KV heads are "
                "replicated across ranks. KV cache does NOT shrink with TP past that point."
            )
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        return "\n".join(lines)


def _activation_estimate(m: ModelSpec, cfg: ServingConfig) -> float:
    """Peak activation bytes during a forward pass, per GPU.

    vLLM measures this empirically with a profile run; we approximate it as the
    largest transient tensors for a batch of `max_num_batched_tokens`:
    the logits tensor (batch x vocab) usually dominates for large vocabularies,
    plus hidden states and MLP intermediates. This is an estimate and is
    labelled as such; the profiled value is authoritative.
    """
    b = DTYPE_BYTES[cfg.weight_dtype if cfg.weight_dtype in ("fp16", "bf16", "fp32") else "bf16"]
    n = cfg.max_num_batched_tokens
    hidden = m.hidden_size
    inter = m.moe_intermediate_size or m.intermediate_size or 4 * hidden
    # Logits are computed for a subset of tokens (one per sequence in decode),
    # but a full prefill chunk can materialise a large intermediate.
    logits = min(n, cfg.max_num_seqs) * m.vocab_size * 4  # fp32 logits
    mlp = n * inter * b * 2 / cfg.tp
    hs = n * hidden * b * 4
    return (logits + mlp + hs) / GIB


def compute_budget(
    model: str | ModelSpec,
    gpu: str | GPUSpec,
    cfg: ServingConfig | None = None,
) -> MemoryBudget:
    """Full memory budget for one GPU in the deployment."""
    m = model if isinstance(model, ModelSpec) else get_model(model)
    g = gpu if isinstance(gpu, GPUSpec) else get_gpu(gpu)
    cfg = cfg or ServingConfig()

    b = MemoryBudget(gpu=g.name, model=m.name)
    b.total_gib = g.vram_bytes / GIB
    b.usable_gib = b.total_gib * cfg.gpu_memory_utilization

    # --- weights, sharded across the whole world ---------------------------
    world = cfg.world_size()
    b.weights_gib = m.weight_bytes(cfg.weight_dtype) / GIB / world

    # --- KV heads per GPU --------------------------------------------------
    # Tensor parallelism shards KV heads. When TP exceeds the number of KV
    # heads (common with GQA: 8 KV heads but TP=16), the heads are replicated
    # instead, so KV cache per GPU stops shrinking. Missing this is how a
    # "just add more GPUs" plan fails to buy any extra context.
    if cfg.tp <= m.num_kv_heads:
        if m.num_kv_heads % cfg.tp != 0:
            b.problems.append(
                f"num_kv_heads={m.num_kv_heads} is not divisible by TP={cfg.tp}; "
                "vLLM will reject this configuration"
            )
        b.kv_heads_per_gpu = max(m.num_kv_heads // cfg.tp, 1)
    else:
        b.kv_heads_per_gpu = 1
        b.kv_heads_replicated = True

    layers_per_gpu = math.ceil(m.num_layers / cfg.pp)
    if m.attn_kind == "mla":
        # MLA's latent cache is not shardable by head; it is replicated.
        per_layer = (m.kv_lora_rank + m.qk_rope_head_dim) * DTYPE_BYTES[cfg.kv_dtype]
        b.kv_bytes_per_token = layers_per_gpu * per_layer
        b.kv_heads_replicated = True
    else:
        b.kv_bytes_per_token = (
            2 * layers_per_gpu * b.kv_heads_per_gpu * m.head_dim * DTYPE_BYTES[cfg.kv_dtype]
        )

    # --- other consumers ---------------------------------------------------
    b.activation_gib = _activation_estimate(m, cfg)
    b.cuda_graph_gib = 1.5 if cfg.enable_cuda_graphs else 0.0
    b.framework_overhead_gib = 0.8 + (0.6 if world > 1 else 0.0)  # CUDA ctx, NCCL buffers

    b.kv_cache_gib = (
        b.usable_gib
        - b.weights_gib
        - b.activation_gib
        - b.cuda_graph_gib
        - b.framework_overhead_gib
    )

    b.block_bytes = cfg.block_size * b.kv_bytes_per_token
    b.num_blocks = int(max(b.kv_cache_gib, 0) * GIB // b.block_bytes) if b.block_bytes else 0
    b.max_cached_tokens = b.num_blocks * cfg.block_size

    # --- feasibility -------------------------------------------------------
    if b.kv_cache_gib <= 0:
        b.fits = False
        b.problems.append(
            f"weights alone ({b.weights_gib:.1f} GiB) plus overhead exceed the "
            f"{b.usable_gib:.1f} GiB budget: no KV cache left. "
            "Increase TP, quantise weights, or use a larger GPU."
        )
    elif b.max_cached_tokens < cfg.max_model_len:
        b.fits = False
        b.problems.append(
            f"max_model_len={cfg.max_model_len:,} but only {b.max_cached_tokens:,} "
            "tokens of KV cache exist: a single full-length request cannot be served. "
            "vLLM will refuse to start."
        )
    elif b.max_cached_tokens < cfg.max_model_len * 2:
        b.problems.append(
            f"only {b.max_cached_tokens / cfg.max_model_len:.1f} full-length sequences "
            "fit. Any concurrency will cause preemption thrash."
        )

    b.sliding_window = m.sliding_window
    if m.sliding_window:
        b.windowed_layer_frac = (
            1.0 if m.layers_with_window is None
            else min(m.layers_with_window / m.num_layers, 1.0)
        )
    b.config = {
        "tp": cfg.tp, "pp": cfg.pp, "world_size": world,
        "weight_dtype": cfg.weight_dtype, "kv_dtype": cfg.kv_dtype,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "block_size": cfg.block_size, "max_model_len": cfg.max_model_len,
        "num_kv_heads": m.num_kv_heads, "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
    }
    return b


def required_gpus(
    model: str | ModelSpec,
    gpu: str | GPUSpec,
    *,
    target_seq_len: int,
    target_concurrency: int,
    cfg: ServingConfig | None = None,
    max_tp: int = 8,
) -> dict[str, Any]:
    """Smallest TP degree that serves the target workload, or why none does."""
    m = model if isinstance(model, ModelSpec) else get_model(model)
    g = gpu if isinstance(gpu, GPUSpec) else get_gpu(gpu)
    base = cfg or ServingConfig()
    attempts = []
    for tp in [t for t in (1, 2, 4, 8, 16) if t <= max_tp]:
        c = ServingConfig(**{**base.__dict__, "tp": tp, "max_model_len": target_seq_len})
        b = compute_budget(m, g, c)
        head = b.headroom_ratio(target_seq_len, target_concurrency)
        attempts.append({
            "tp": tp, "fits": b.fits and head >= 1.0, "headroom": head,
            "kv_gib": round(b.kv_cache_gib, 2),
            "max_concurrent": b.max_concurrent_seqs(target_seq_len),
            "problems": list(b.problems),
        })
        if b.fits and head >= 1.0:
            return {"ok": True, "tp": tp, "budget": b, "attempts": attempts}
    return {"ok": False, "tp": None, "budget": None, "attempts": attempts}
