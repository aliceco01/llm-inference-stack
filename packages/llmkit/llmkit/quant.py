"""Quantization schemes and what each one actually costs.

The distinction that decides everything, and that most comparisons blur:

**Weight-only quantization** (AWQ, GPTQ, W4A16) stores weights at 4 bits and
dequantizes them to fp16 inside the kernel before a normal fp16 matmul. Memory
traffic drops 4x, so memory-bound *decode* gets much faster. Arithmetic is
unchanged and there is extra dequant work per element, so compute-bound
*prefill* gets slightly SLOWER. Net effect depends entirely on your
input/output ratio: great for chat, bad for long-prompt summarisation.

**Weight and activation quantization** (FP8, INT8 W8A8) quantizes both operands
and uses native low-precision tensor cores. Memory traffic halves AND FLOPs
double on hardware that supports it (Hopper fp8, Ada fp8, Ampere int8). Both
phases get faster. The cost is accuracy: activations have outliers that weights
do not, which is why INT8 needs SmoothQuant-style calibration and FP8 mostly
does not (its wider dynamic range absorbs them).

**KV cache quantization is a separate axis** from weight quantization and is
frequently the bigger win for long-context serving, because at 128k context the
KV cache dwarfs the weights. You can run bf16 weights with fp8 KV, and often
should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .gpus import GPUSpec
from .modelspec import ModelSpec

QuantKind = Literal["weight_only", "weight_activation", "none"]


@dataclass
class QuantScheme:
    """One quantization configuration, with the properties that predict its
    behaviour rather than just its name."""

    name: str
    weight_bits: float
    activation_bits: float
    kind: QuantKind
    kv_dtype: str = "fp16"

    # Kernel realities, not marketing numbers.
    needs_calibration: bool = False
    dequant_overhead: float = 0.0     # fraction added to compute-bound work
    tensor_core_speedup: float = 1.0  # FLOPs multiplier vs bf16 on supporting HW
    requires_compute_capability: float = 0.0   # 8.9 = Ada, 9.0 = Hopper
    vllm_flag: str = ""
    notes: str = ""

    @property
    def weight_bytes_per_param(self) -> float:
        # 4-bit schemes carry per-group scales and zero points. With group size
        # 128 that is roughly 0.2 extra bits per weight, which is small but not
        # zero, and ignoring it under-predicts model size by a few percent.
        overhead = 0.25 if self.weight_bits <= 4 else 0.0
        return (self.weight_bits + overhead) / 8.0

    def supported_on(self, gpu: GPUSpec) -> bool:
        if not self.requires_compute_capability:
            return True
        cc = _COMPUTE_CAPABILITY.get(gpu.name, 0.0)
        return cc >= self.requires_compute_capability


_COMPUTE_CAPABILITY = {
    "a10g": 8.6, "l4": 8.9, "l40s": 8.9, "rtx4090": 8.9,
    "a100-40gb": 8.0, "a100-80gb": 8.0,
    "h100-pcie": 9.0, "h100-sxm": 9.0, "h200": 9.0, "b200": 10.0,
    "mi300x": 0.0,
}


SCHEMES: dict[str, QuantScheme] = {
    s.name: s
    for s in [
        QuantScheme("bf16", 16, 16, "none", "fp16",
                    vllm_flag="--dtype bfloat16",
                    notes="baseline; every other number is relative to this"),
        QuantScheme("fp8", 8, 8, "weight_activation", "fp8",
                    tensor_core_speedup=2.0, requires_compute_capability=8.9,
                    vllm_flag="--quantization fp8 --kv-cache-dtype fp8",
                    notes="Hopper/Ada native. Usually the best default: near-lossless, "
                          "halves weights and KV, doubles FLOPs, needs no calibration."),
        QuantScheme("int8-w8a8", 8, 8, "weight_activation", "fp16",
                    needs_calibration=True, tensor_core_speedup=2.0,
                    requires_compute_capability=8.0,
                    vllm_flag="--quantization compressed-tensors",
                    notes="SmoothQuant-style. Works on Ampere where fp8 does not. "
                          "Activation outliers make calibration mandatory."),
        QuantScheme("awq-int4", 4, 16, "weight_only", "fp16",
                    needs_calibration=True, dequant_overhead=0.30,
                    requires_compute_capability=7.5,
                    vllm_flag="--quantization awq_marlin",
                    notes="W4A16. Big decode win, prefill regression. "
                          "Activation-aware scaling protects salient channels."),
        QuantScheme("gptq-int4", 4, 16, "weight_only", "fp16",
                    needs_calibration=True, dequant_overhead=0.35,
                    requires_compute_capability=7.5,
                    vllm_flag="--quantization gptq_marlin",
                    notes="W4A16, second-order weight rounding. Similar profile to AWQ."),
        QuantScheme("fp4", 4, 4, "weight_activation", "fp8",
                    tensor_core_speedup=4.0, requires_compute_capability=10.0,
                    vllm_flag="--quantization modelopt_fp4",
                    notes="Blackwell only. Both operands at 4 bits."),
        QuantScheme("bf16-kv8", 16, 16, "none", "fp8",
                    vllm_flag="--kv-cache-dtype fp8",
                    notes="KV quantization alone. Full-precision weights, half the KV. "
                          "Often the highest value-per-risk change for long context."),
    ]
}


@dataclass
class QuantPrediction:
    """Predicted effect of a scheme, before you spend a GPU-hour measuring."""

    scheme: str
    supported: bool = True
    unsupported_reason: str = ""

    weights_gib: float = 0.0
    kv_gib_per_1k_tokens: float = 0.0
    kv_cache_gib: float = 0.0
    max_cached_tokens: int = 0

    decode_speedup: float = 1.0
    prefill_speedup: float = 1.0
    notes: str = ""
    risks: list[str] = field(default_factory=list)


def predict(model: ModelSpec, gpu: GPUSpec, scheme: QuantScheme, *,
            tp: int = 1, gpu_memory_utilization: float = 0.90,
            batch_size: int = 32, ctx_len: int = 2048) -> QuantPrediction:
    """Predict memory and speed for a scheme using the roofline model.

    This is a prediction, not a measurement, and it says nothing about quality.
    Its job is to rank candidates so you measure the two that matter instead of
    all six.
    """
    from .kvcache import ServingConfig, compute_budget

    p = QuantPrediction(scheme=scheme.name, notes=scheme.notes)
    if not scheme.supported_on(gpu):
        p.supported = False
        need = scheme.requires_compute_capability
        p.unsupported_reason = (
            f"{scheme.name} needs compute capability {need}; "
            f"{gpu.name} is {_COMPUTE_CAPABILITY.get(gpu.name, 0.0)}"
        )
        return p

    # --- memory ---------------------------------------------------------
    weight_bytes = model.params_b * 1e9 * scheme.weight_bytes_per_param / tp
    p.weights_gib = weight_bytes / 1024 ** 3

    kv_per_tok = model.kv_bytes_per_token(scheme.kv_dtype) / max(tp, 1)
    p.kv_gib_per_1k_tokens = kv_per_tok * 1000 / 1024 ** 3

    cfg = ServingConfig(tp=tp, kv_dtype=scheme.kv_dtype,
                        gpu_memory_utilization=gpu_memory_utilization,
                        weight_dtype="bf16" if scheme.weight_bits >= 16 else "fp8")
    budget = compute_budget(model, gpu, cfg)
    # Override the weight term with this scheme's real footprint.
    delta = budget.weights_gib - p.weights_gib
    p.kv_cache_gib = max(budget.kv_cache_gib + delta, 0.0)
    block_bytes = cfg.block_size * kv_per_tok
    p.max_cached_tokens = int(p.kv_cache_gib * 1024 ** 3 // block_bytes) * cfg.block_size if block_bytes else 0

    # --- speed ----------------------------------------------------------
    # Decode is memory bound: speedup tracks the reduction in bytes read per
    # step, which is weights plus KV for the running batch.
    base_w = model.params_b * 1e9 * 2 / tp
    base_kv = model.kv_bytes_per_token("fp16") / max(tp, 1) * batch_size * ctx_len
    new_kv = kv_per_tok * batch_size * ctx_len
    p.decode_speedup = (base_w + base_kv) / max(weight_bytes + new_kv, 1.0)

    # Prefill is compute bound.
    if scheme.kind == "weight_activation":
        p.prefill_speedup = scheme.tensor_core_speedup
    elif scheme.kind == "weight_only":
        # No FLOPs saved, and dequant adds work. This is the counterintuitive
        # result that decides whether AWQ helps your workload.
        p.prefill_speedup = 1.0 / (1.0 + scheme.dequant_overhead)
    else:
        p.prefill_speedup = 1.0

    # --- risks ----------------------------------------------------------
    if scheme.needs_calibration:
        p.risks.append(
            "requires a calibration set; quality depends on how well it matches "
            "your traffic, so calibrating on wikitext and serving code is a real risk"
        )
    if scheme.kind == "weight_only":
        p.risks.append(
            f"prefill is {1/p.prefill_speedup:.2f}x SLOWER (dequant overhead with no "
            "FLOP reduction); a bad trade for long-prompt workloads"
        )
    if scheme.weight_bits <= 4:
        p.risks.append(
            "4-bit weights degrade measurably on multi-step reasoning and long "
            "generations even when short-prompt perplexity looks fine"
        )
    if scheme.kv_dtype == "fp8":
        p.risks.append(
            "fp8 KV cache is usually near-lossless but interacts with long context; "
            "measure at your actual p95 context length, not at 512 tokens"
        )
    return p


def compare(model: ModelSpec, gpu: GPUSpec, *, tp: int = 1,
            batch_size: int = 32, ctx_len: int = 2048,
            schemes: list[str] | None = None) -> list[QuantPrediction]:
    names = schemes or list(SCHEMES)
    return [predict(model, gpu, SCHEMES[n], tp=tp, batch_size=batch_size,
                    ctx_len=ctx_len) for n in names]
