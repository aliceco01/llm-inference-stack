"""Model architecture facts needed for memory math.

The registry below is a convenience. The authoritative path is
`ModelSpec.from_hf_config()`, which reads the model's own config.json, because
a hardcoded table is exactly the kind of thing that silently goes stale and
produces a confidently wrong VRAM estimate.

The field that matters most is `num_kv_heads`. Grouped-query attention makes it
much smaller than `num_attention_heads` (8 vs 64 on Llama-3-70B), and sizing KV
cache off attention heads instead of KV heads overestimates by 8x. That single
mistake is the most common source of "why is my KV cache math wrong".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4", "fp4"]

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
}

AttnKind = Literal["mha", "gqa", "mqa", "mla"]


@dataclass
class ModelSpec:
    """Architecture facts that determine memory footprint."""

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int = 0
    vocab_size: int = 32000
    intermediate_size: int = 0
    params_b: float = 0.0            # total parameters in billions
    max_position: int = 8192
    tie_word_embeddings: bool = False

    # sliding window / local attention caps KV growth past `window`
    sliding_window: int | None = None
    layers_with_window: int | None = None   # None = all layers

    # Mixture of experts. Experts usually have their own (smaller)
    # intermediate size; using the dense `intermediate_size` for them
    # overestimates MoE parameter counts by an order of magnitude.
    n_experts: int = 0
    n_active_experts: int = 0
    moe_intermediate_size: int = 0
    n_shared_experts: int = 0
    first_k_dense: int = 0          # leading layers that are dense, not MoE

    # Multi-head latent attention (DeepSeek-style): a single compressed cache
    # per layer instead of separate K and V tensors.
    attn_kind: AttnKind = "gqa"
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0

    source: str = "registry"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.head_dim:
            self.head_dim = self.hidden_size // max(self.num_attention_heads, 1)
        if self.num_kv_heads == self.num_attention_heads and self.attn_kind == "gqa":
            self.attn_kind = "mha"
        if self.num_kv_heads == 1:
            self.attn_kind = "mqa"

    # ------------------------------------------------------------------
    @property
    def gqa_ratio(self) -> float:
        return self.num_attention_heads / max(self.num_kv_heads, 1)

    def kv_bytes_per_token(self, kv_dtype: DType = "fp16") -> float:
        """Bytes of KV cache consumed by ONE token across ALL layers.

        Standard attention stores K and V separately, hence the factor 2:
            2 * layers * kv_heads * head_dim * dtype_bytes

        MLA stores one compressed latent vector plus a small RoPE component,
        with no factor of 2, which is why DeepSeek-V3 has a smaller KV cache
        per token than a 7B GQA model despite being far larger.
        """
        b = DTYPE_BYTES[kv_dtype]
        if self.attn_kind == "mla":
            per_layer = (self.kv_lora_rank + self.qk_rope_head_dim) * b
            return self.num_layers * per_layer
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * b

    def kv_bytes_for_sequence(self, seq_len: int, kv_dtype: DType = "fp16") -> float:
        """KV bytes for a sequence of `seq_len` tokens, honouring sliding window.

        With sliding-window attention the cache stops growing at the window
        size, so a 128k-context Mistral-style model costs far less than the
        naive length x per-token product suggests.
        """
        per_tok = self.kv_bytes_per_token(kv_dtype)
        if not self.sliding_window:
            return per_tok * seq_len

        eff = min(seq_len, self.sliding_window)
        if self.layers_with_window is None or self.layers_with_window >= self.num_layers:
            return per_tok * eff
        # Mixed: some layers windowed, the rest full.
        frac_w = self.layers_with_window / self.num_layers
        return per_tok * (frac_w * eff + (1 - frac_w) * seq_len)

    def weight_bytes(self, dtype: DType = "fp16") -> float:
        """Total weight bytes.

        For MoE models all experts are resident even though only a few are
        active per token, so memory tracks total parameters while compute
        tracks active parameters. Conflating the two is how people conclude a
        Mixtral-8x7B "is a 13B model" and then OOM.
        """
        if self.params_b <= 0:
            return self.estimate_params() * DTYPE_BYTES[dtype]
        return self.params_b * 1e9 * DTYPE_BYTES[dtype]

    def active_params(self) -> float:
        if not (self.n_experts and self.n_active_experts):
            return self.params_b * 1e9
        moe_layers = max(self.num_layers - self.first_k_dense, 0)
        e_inter = self.moe_intermediate_size or self.intermediate_size or 4 * self.hidden_size
        per_expert = 3 * self.hidden_size * e_inter
        total_expert = moe_layers * per_expert * self.n_experts
        active_expert = moe_layers * per_expert * (self.n_active_experts + self.n_shared_experts)
        return self.params_b * 1e9 - total_expert + active_expert

    def estimate_params(self) -> float:
        """Parameter count from architecture, when the registry lacks it."""
        h, L = self.hidden_size, self.num_layers
        inter = self.intermediate_size or 4 * h
        q = h * (self.num_attention_heads * self.head_dim)
        kv = 2 * h * (self.num_kv_heads * self.head_dim)
        o = (self.num_attention_heads * self.head_dim) * h
        attn = q + kv + o
        dense_ffn = 3 * h * inter    # gated MLP: gate, up, down
        if self.n_experts:
            e_inter = self.moe_intermediate_size or inter
            moe_ffn = 3 * h * e_inter * (self.n_experts + self.n_shared_experts)
            n_dense = min(self.first_k_dense, L)
            total_ffn = n_dense * dense_ffn + (L - n_dense) * moe_ffn
        else:
            total_ffn = L * dense_ffn
        emb = self.vocab_size * h * (1 if self.tie_word_embeddings else 2)
        return L * (attn + 2 * h) + total_ffn + emb + h

    # ------------------------------------------------------------------
    @classmethod
    def from_hf_config(cls, path_or_dict: str | Path | dict[str, Any], name: str = "") -> ModelSpec:
        """Build from a Hugging Face config.json. This is the trustworthy path."""
        if isinstance(path_or_dict, (str, Path)):
            p = Path(path_or_dict)
            if p.is_dir():
                p = p / "config.json"
            cfg = json.loads(p.read_text())
            src = str(p)
        else:
            cfg = dict(path_or_dict)
            src = "dict"

        text_cfg = cfg.get("text_config") or cfg  # multimodal wrappers
        n_heads = int(text_cfg.get("num_attention_heads", 32))
        n_kv = int(text_cfg.get("num_key_value_heads", n_heads))
        hidden = int(text_cfg.get("hidden_size", 4096))
        head_dim = int(text_cfg.get("head_dim") or (hidden // max(n_heads, 1)))

        kv_lora = int(text_cfg.get("kv_lora_rank") or 0)
        attn_kind: AttnKind = "mla" if kv_lora else ("mqa" if n_kv == 1 else
                                                     ("mha" if n_kv == n_heads else "gqa"))
        sw = text_cfg.get("sliding_window")
        spec = cls(
            name=name or cfg.get("_name_or_path") or "from-config",
            num_layers=int(text_cfg.get("num_hidden_layers", 32)),
            hidden_size=hidden,
            num_attention_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            vocab_size=int(text_cfg.get("vocab_size", 32000)),
            intermediate_size=int(text_cfg.get("intermediate_size", 0)),
            max_position=int(text_cfg.get("max_position_embeddings", 8192)),
            tie_word_embeddings=bool(text_cfg.get("tie_word_embeddings", False)),
            sliding_window=int(sw) if sw else None,
            n_experts=int(text_cfg.get("num_local_experts") or text_cfg.get("n_routed_experts") or 0),
            n_active_experts=int(text_cfg.get("num_experts_per_tok") or 0),
            moe_intermediate_size=int(text_cfg.get("moe_intermediate_size") or 0),
            n_shared_experts=int(text_cfg.get("n_shared_experts") or 0),
            first_k_dense=int(text_cfg.get("first_k_dense_replace") or 0),
            attn_kind=attn_kind,
            kv_lora_rank=kv_lora,
            qk_rope_head_dim=int(text_cfg.get("qk_rope_head_dim") or 0),
            source=src,
        )
        spec.params_b = spec.estimate_params() / 1e9
        return spec

    def describe(self) -> str:
        return (
            f"{self.name}: {self.num_layers}L h={self.hidden_size} "
            f"heads={self.num_attention_heads}/kv={self.num_kv_heads} "
            f"({self.attn_kind}, GQA ratio {self.gqa_ratio:.0f}x) "
            f"head_dim={self.head_dim} params={self.params_b:.1f}B"
            + (f" window={self.sliding_window}" if self.sliding_window else "")
        )


# ---------------------------------------------------------------------------
# Convenience registry. Values follow each model's published config.json.
# Treat as a starting point; from_hf_config() is authoritative.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("llama-3.2-1b", 16, 2048, 32, 8, head_dim=64, vocab_size=128256,
                  intermediate_size=8192, params_b=1.24, max_position=131072,
                  tie_word_embeddings=True),
        ModelSpec("llama-3.2-3b", 28, 3072, 24, 8, head_dim=128, vocab_size=128256,
                  intermediate_size=8192, params_b=3.21, max_position=131072,
                  tie_word_embeddings=True),
        ModelSpec("llama-3.1-8b", 32, 4096, 32, 8, head_dim=128, vocab_size=128256,
                  intermediate_size=14336, params_b=8.03, max_position=131072),
        ModelSpec("llama-3.1-70b", 80, 8192, 64, 8, head_dim=128, vocab_size=128256,
                  intermediate_size=28672, params_b=70.6, max_position=131072),
        ModelSpec("llama-3.1-405b", 126, 16384, 128, 8, head_dim=128, vocab_size=128256,
                  intermediate_size=53248, params_b=405.9, max_position=131072),
        ModelSpec("mistral-7b-v0.3", 32, 4096, 32, 8, head_dim=128, vocab_size=32768,
                  intermediate_size=14336, params_b=7.25, max_position=32768),
        ModelSpec("mixtral-8x7b", 32, 4096, 32, 8, head_dim=128, vocab_size=32000,
                  intermediate_size=14336, params_b=46.7, max_position=32768,
                  n_experts=8, n_active_experts=2),
        ModelSpec("qwen2.5-7b", 28, 3584, 28, 4, head_dim=128, vocab_size=152064,
                  intermediate_size=18944, params_b=7.62, max_position=32768),
        ModelSpec("qwen2.5-32b", 64, 5120, 40, 8, head_dim=128, vocab_size=152064,
                  intermediate_size=27648, params_b=32.8, max_position=32768),
        ModelSpec("qwen2.5-72b", 80, 8192, 64, 8, head_dim=128, vocab_size=152064,
                  intermediate_size=29568, params_b=72.7, max_position=32768),
        ModelSpec("phi-3-mini-4k", 32, 3072, 32, 32, head_dim=96, vocab_size=32064,
                  intermediate_size=8192, params_b=3.82, max_position=4096),
        ModelSpec("gemma-2-9b", 42, 3584, 16, 8, head_dim=256, vocab_size=256000,
                  intermediate_size=14336, params_b=9.24, max_position=8192,
                  sliding_window=4096, layers_with_window=21, tie_word_embeddings=True,
                  notes="alternating local/global attention: half the layers are windowed"),
        ModelSpec("deepseek-v3", 61, 7168, 128, 128, head_dim=192, vocab_size=129280,
                  intermediate_size=18432, params_b=671.0, max_position=163840,
                  n_experts=256, n_active_experts=8, moe_intermediate_size=2048,
                  n_shared_experts=1, first_k_dense=3,
                  attn_kind="mla", kv_lora_rank=512, qk_rope_head_dim=64,
                  notes="MLA: compressed latent KV, single cache, no K/V factor of 2"),
    ]
}


def get_model(name: str) -> ModelSpec:
    key = name.lower().strip()
    if key in REGISTRY:
        return REGISTRY[key]
    # tolerate HF-style ids like meta-llama/Llama-3.1-8B-Instruct
    short = key.split("/")[-1].replace("-instruct", "").replace("meta-", "")
    for k in REGISTRY:
        if k in short or short in k:
            return REGISTRY[k]
    raise KeyError(
        f"unknown model {name!r}. Known: {', '.join(sorted(REGISTRY))}. "
        "For anything else use ModelSpec.from_hf_config(path/to/config.json)."
    )
