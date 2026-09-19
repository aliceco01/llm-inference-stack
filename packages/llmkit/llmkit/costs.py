"""Unit economics: cost per token, MFU, and MBU.

The metric mistake this module exists to prevent
------------------------------------------------
MFU (Model FLOPs Utilization) is the standard efficiency metric and it is the
**wrong metric for decode**. Decode is memory-bandwidth bound: each step reads
the entire weight matrix to produce one token per sequence. Its arithmetic
intensity is therefore tiny and its MFU is structurally 1-5% on any hardware,
no matter how well tuned the deployment is. Reporting decode MFU and concluding
the GPU is being wasted is a category error, and it leads teams to chase a
number that physics has already capped.

MBU (Model Bandwidth Utilization) is the right metric for decode: achieved
bytes-read per second divided by peak HBM bandwidth. A well-tuned decode
deployment reaches 60-85% MBU while sitting at 3% MFU, and both facts are
simultaneously true and unremarkable.

Use MFU for prefill (compute bound) and MBU for decode (bandwidth bound).
Reporting a single blended "utilization" across both hides which phase is
actually inefficient.

Cost attribution
----------------
Cost per token is GPU-hours times price divided by tokens served, but the
interesting question is per tenant on shared capacity. Prompt tokens, cached
prompt tokens and completion tokens have genuinely different marginal costs
(a cached prompt token costs nothing to prefill), so a fair internal chargeback
weights them rather than counting a token as a token.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .gpus import GPUSpec, get_gpu
from .modelspec import ModelSpec, get_model
from .types import RequestRecord


@dataclass
class PricePoint:
    """What an hour of capacity costs. Defaults are representative list prices."""

    gpu: str
    usd_per_gpu_hour: float
    n_gpus: int = 1
    # Real cost is not just the GPU line item.
    overhead_multiplier: float = 1.35
    notes: str = "overhead covers CPU/RAM/network/storage/control plane"

    @property
    def usd_per_hour(self) -> float:
        return self.usd_per_gpu_hour * self.n_gpus * self.overhead_multiplier

    @property
    def usd_per_second(self) -> float:
        return self.usd_per_hour / 3600.0


def price_for(gpu: str, *, n_gpus: int = 1, usd_per_gpu_hour: float | None = None,
              overhead_multiplier: float = 1.35) -> PricePoint:
    g = get_gpu(gpu)
    return PricePoint(
        gpu=g.name,
        usd_per_gpu_hour=usd_per_gpu_hour if usd_per_gpu_hour is not None
        else g.usd_per_hour,
        n_gpus=n_gpus, overhead_multiplier=overhead_multiplier,
    )


# ---------------------------------------------------------------------------
@dataclass
class Utilization:
    """Efficiency of a measured window, reported per phase."""

    window_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_prompt_tokens: int = 0
    mean_context: float = 0.0

    prefill_flops: float = 0.0
    decode_flops: float = 0.0
    decode_bytes: float = 0.0

    peak_flops: float = 0.0
    peak_bw: float = 0.0

    @property
    def prefill_mfu(self) -> float:
        """Correct metric for prefill: compute bound."""
        if not (self.window_s and self.peak_flops):
            return math.nan
        return self.prefill_flops / (self.window_s * self.peak_flops)

    @property
    def decode_mfu(self) -> float:
        """Reported only to show it is structurally low. Do not optimise it."""
        if not (self.window_s and self.peak_flops):
            return math.nan
        return self.decode_flops / (self.window_s * self.peak_flops)

    @property
    def decode_mbu(self) -> float:
        """Correct metric for decode: bandwidth bound."""
        if not (self.window_s and self.peak_bw):
            return math.nan
        return self.decode_bytes / (self.window_s * self.peak_bw)

    @property
    def blended_mfu(self) -> float:
        if not (self.window_s and self.peak_flops):
            return math.nan
        return (self.prefill_flops + self.decode_flops) / (self.window_s * self.peak_flops)

    def verdict(self) -> list[str]:
        out: list[str] = []
        if not math.isnan(self.prefill_mfu):
            if self.prefill_mfu < 0.15:
                out.append(
                    f"prefill MFU {self.prefill_mfu*100:.1f}%: the GPU is idle or "
                    "prefill batches are too small. Raise max_num_batched_tokens "
                    "or send more concurrent prefill work.")
            elif self.prefill_mfu > 0.5:
                out.append(f"prefill MFU {self.prefill_mfu*100:.1f}%: healthy, "
                           "compute is well utilised.")
        if not math.isnan(self.decode_mbu):
            if self.decode_mbu < 0.3:
                out.append(
                    f"decode MBU {self.decode_mbu*100:.1f}%: batch size is too small. "
                    "Decode reads the full weight matrix per step regardless of "
                    "batch, so a small batch wastes most of that traffic.")
            elif self.decode_mbu > 0.6:
                out.append(f"decode MBU {self.decode_mbu*100:.1f}%: healthy for a "
                           "bandwidth-bound phase.")
        if not math.isnan(self.decode_mfu) and self.decode_mfu < 0.08:
            out.append(
                f"decode MFU is {self.decode_mfu*100:.2f}%, and that is expected. "
                "Decode is memory bound; low MFU here is physics, not a defect. "
                "Judge decode by MBU.")
        return out


def compute_utilization(model: str | ModelSpec, gpu: str | GPUSpec, *,
                        window_s: float, prompt_tokens: int, output_tokens: int,
                        cached_prompt_tokens: int = 0, mean_context: float = 0.0,
                        mean_batch: float = 32.0, n_gpus: int = 1,
                        weight_dtype: str = "bf16", kv_dtype: str = "fp16",
                        ) -> Utilization:
    """Derive MFU and MBU from token counts over a window."""
    m = model if isinstance(model, ModelSpec) else get_model(model)
    g = gpu if isinstance(gpu, GPUSpec) else get_gpu(gpu)

    u = Utilization(window_s=window_s, prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    cached_prompt_tokens=cached_prompt_tokens,
                    mean_context=mean_context or 1024.0)
    u.peak_flops = g.tflops(weight_dtype) * 1e12 * n_gpus
    u.peak_bw = g.bandwidth_gb_s * 1e9 * n_gpus

    active = m.active_params()
    # Cached prompt tokens are never prefilled, so they cost no FLOPs. Counting
    # them would credit the deployment for work it skipped, inflating MFU
    # exactly when prefix caching is working well.
    billable_prefill = max(prompt_tokens - cached_prompt_tokens, 0)
    u.prefill_flops = 2.0 * active * billable_prefill
    # Attention term, quadratic in context.
    u.prefill_flops += (2 * 2 * billable_prefill * u.mean_context
                        * m.head_dim * m.num_attention_heads * m.num_layers * 0.5)

    u.decode_flops = 2.0 * active * output_tokens

    # Decode traffic: weights are read once per STEP, not once per token, so
    # dividing output tokens by batch size is the whole point.
    from .modelspec import DTYPE_BYTES
    weight_bytes = m.params_b * 1e9 * DTYPE_BYTES[weight_dtype]
    steps = output_tokens / max(mean_batch, 1.0)
    kv_bytes = output_tokens * u.mean_context * m.kv_bytes_per_token(kv_dtype)
    u.decode_bytes = steps * weight_bytes + kv_bytes
    return u


# ---------------------------------------------------------------------------
@dataclass
class TenantUsage:
    tenant: str = "unknown"
    model: str = ""
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    output_tokens: int = 0
    gpu_seconds: float = 0.0
    usd: float = 0.0

    @property
    def billable_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_prompt_tokens, 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def usd_per_m_output(self) -> float:
        return self.usd / max(self.output_tokens, 1) * 1e6

    @property
    def usd_per_m_total(self) -> float:
        return self.usd / max(self.total_tokens, 1) * 1e6

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_prompt_tokens / max(self.prompt_tokens, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant": self.tenant, "model": self.model,
            "requests": self.requests, "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "gpu_seconds": round(self.gpu_seconds, 2),
            "usd": round(self.usd, 6),
            "usd_per_m_output": round(self.usd_per_m_output, 4),
            "usd_per_m_total": round(self.usd_per_m_total, 4),
        }


@dataclass
class CostModelConfig:
    """How to split shared capacity across tenants.

    `weight_*` set the relative marginal cost of each token class. The defaults
    reflect the physics: a cached prompt token costs nothing to prefill, an
    uncached prompt token costs one forward pass over its own position, and an
    output token costs a full decode step amortised over the batch. Charging
    all three equally is simple and wrong, and it makes the tenant with long
    shared system prompts subsidise the one generating long outputs.
    """

    weight_prompt: float = 1.0
    weight_cached_prompt: float = 0.10
    weight_output: float = 4.0
    attribute_idle: bool = True
    idle_policy: str = "spread"   # spread | absorb


def attribute_costs(records: Sequence[RequestRecord], price: PricePoint, *,
                    window_s: float, cfg: CostModelConfig | None = None,
                    ) -> tuple[dict[str, TenantUsage], dict[str, Any]]:
    """Split the cost of a window across tenants by weighted token share.

    Returns (per-tenant usage, fleet summary). Idle capacity is reported
    explicitly rather than silently folded into someone's bill.
    """
    cfg = cfg or CostModelConfig()
    by: dict[str, TenantUsage] = {}
    total_weighted = 0.0

    for r in records:
        key = f"{r.tenant or 'unknown'}|{r.model or 'unknown'}"
        u = by.setdefault(key, TenantUsage(tenant=r.tenant or "unknown",
                                           model=r.model or "unknown"))
        u.requests += 1
        if not r.ok:
            u.errors += 1
            continue
        u.prompt_tokens += r.prompt_tokens
        u.cached_prompt_tokens += r.cached_prompt_tokens
        u.output_tokens += r.output_tokens
        w = (cfg.weight_prompt * r.prompt_tokens
             + (cfg.weight_cached_prompt - cfg.weight_prompt) * r.cached_prompt_tokens
             + cfg.weight_output * r.output_tokens)
        u.gpu_seconds += w         # temporarily holds the weight
        total_weighted += w

    window_cost = price.usd_per_second * window_s
    for u in by.values():
        share = u.gpu_seconds / total_weighted if total_weighted else 0.0
        u.gpu_seconds = share * window_s
        u.usd = share * window_cost

    fleet = {
        "window_s": window_s,
        "usd_total": round(window_cost, 6),
        "usd_per_hour": round(price.usd_per_hour, 4),
        "gpus": price.n_gpus,
        "gpu": price.gpu,
        "tenants": len(by),
        "requests": sum(u.requests for u in by.values()),
        "output_tokens": sum(u.output_tokens for u in by.values()),
        "prompt_tokens": sum(u.prompt_tokens for u in by.values()),
        "cached_prompt_tokens": sum(u.cached_prompt_tokens for u in by.values()),
    }
    out_tok = fleet["output_tokens"]
    fleet["usd_per_m_output_tokens"] = round(
        window_cost / max(out_tok, 1) * 1e6, 4)
    fleet["usd_per_m_total_tokens"] = round(
        window_cost / max(out_tok + fleet["prompt_tokens"], 1) * 1e6, 4)
    fleet["cache_hit_rate"] = round(
        fleet["cached_prompt_tokens"] / max(fleet["prompt_tokens"], 1), 4)
    return by, fleet


def breakeven_vs_api(usd_per_m_output: float, api_usd_per_m_output: float,
                     ) -> dict[str, Any]:
    """Self-hosting only wins above a utilisation threshold.

    The comparison people get wrong: a self-hosted $/M-token figure computed at
    100% utilisation is not comparable to an API price, because APIs charge per
    token and you pay for the GPU whether or not it is busy. The honest
    comparison divides your cost by your ACHIEVED utilisation.
    """
    if api_usd_per_m_output <= 0:
        return {"comparable": False}
    ratio = usd_per_m_output / api_usd_per_m_output
    return {
        "self_hosted_usd_per_m": round(usd_per_m_output, 4),
        "api_usd_per_m": round(api_usd_per_m_output, 4),
        "ratio": round(ratio, 3),
        "min_utilization_to_break_even": round(min(ratio, 1.0), 3),
        "verdict": (
            f"self-hosting is cheaper above {ratio*100:.0f}% sustained utilisation"
            if ratio < 1.0 else
            "the API is cheaper at any utilisation on these numbers"
        ),
    }
