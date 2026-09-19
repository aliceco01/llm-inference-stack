"""Workload generation with controlled shapes.

Prefill cost scales with prompt length and decode cost scales with output
length, and they hit different hardware limits (prefill is compute bound,
decode is memory-bandwidth bound). A benchmark that does not control both
independently cannot attribute a latency change to a cause, so every generator
here takes explicit input/output length distributions.

Prompt text is built from a fixed word list with a seeded RNG, so a run is
reproducible and a shared prefix is byte-identical across requests, which is
what prefix caching actually keys on.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from .client import Request

# A deterministic vocabulary. Common short English words tokenise to roughly
# one token each on Llama/GPT BPE vocabularies, which keeps the requested
# token length close to the realised one without needing the vocab locally.
_WORDS = ["time", "year", "people", "way", "day", "man", "thing", "woman", "life", "child", "world", "school", "state", "family", "student", "group", "country", "problem", "hand", "part", "place", "case", "week", "company", "system", "program", "question", "work", "night", "point", "home", "water", "room", "mother", "area", "money", "story", "fact", "month", "lot", "right", "study", "book", "eye", "job", "word", "business", "issue", "side", "kind", "head", "house", "service", "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car", "city", "community", "name", "president", "team", "minute", "idea", "kid", "body", "back", "parent", "face", "level", "office", "door", "health", "person", "art", "war", "history", "party", "result", "change", "morning", "reason", "research", "girl", "guy", "moment", "air", "teacher", "force"]


def make_text(n_tokens: int, seed: int) -> str:
    """Deterministic pseudo-text of approximately `n_tokens` tokens."""
    rng = random.Random(seed)
    n = max(1, n_tokens)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


@dataclass
class LengthSpec:
    """Distribution for one length dimension."""

    kind: Literal["fixed", "uniform", "lognormal"] = "fixed"
    value: int = 512           # fixed value, or the mean for lognormal
    low: int = 0
    high: int = 0
    sigma: float = 0.6         # lognormal shape; 0.6 approximates ShareGPT
    min_v: int = 8
    max_v: int = 32768

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            v = self.value
        elif self.kind == "uniform":
            v = rng.randint(self.low or self.value, self.high or self.value)
        else:
            mu = math.log(max(self.value, 1)) - 0.5 * self.sigma ** 2
            v = int(round(rng.lognormvariate(mu, self.sigma)))
        return max(self.min_v, min(self.max_v, v))


@dataclass
class WorkloadSpec:
    """A complete, reproducible description of what to send."""

    name: str = "synthetic"
    input_len: LengthSpec = field(default_factory=lambda: LengthSpec("fixed", 512))
    output_len: LengthSpec = field(default_factory=lambda: LengthSpec("fixed", 128))
    system_prompt_tokens: int = 0     # shared prefix length, 0 to disable
    n_prefix_variants: int = 1        # distinct system prompts (tenants/personas)
    multi_turn: int = 1               # turns per session; >1 grows the prefix
    seed: int = 1234
    tenants: tuple[str, ...] = ("default",)

    def fingerprint(self) -> str:
        """Stable id so a result file can be tied back to its exact workload."""
        raw = (
            f"{self.name}|{self.input_len}|{self.output_len}|"
            f"{self.system_prompt_tokens}|{self.n_prefix_variants}|"
            f"{self.multi_turn}|{self.seed}|{self.tenants}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


class WorkloadGenerator:
    """Produces Requests. Stateless across calls except for the RNG stream."""

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self._systems = [
            make_text(spec.system_prompt_tokens, seed=spec.seed * 7919 + i)
            for i in range(max(1, spec.n_prefix_variants))
        ] if spec.system_prompt_tokens > 0 else []
        self._counter = 0

    def reset(self) -> None:
        self.rng = random.Random(self.spec.seed)
        self._counter = 0

    def one(self) -> Request:
        return next(iter(self.session(1)))

    def session(self, turns: int | None = None) -> Iterator[Request]:
        """One conversation. Turn k carries the full history, so the shared
        prefix grows monotonically, which is exactly the pattern prefix caching
        is meant to exploit."""
        sp = self.spec
        turns = turns if turns is not None else sp.multi_turn
        sid = f"s{self._counter}"
        self._counter += 1
        tenant = self.rng.choice(sp.tenants)
        system = self._systems[self.rng.randrange(len(self._systems))] if self._systems else None
        sys_tokens = sp.system_prompt_tokens if system else 0

        history = ""
        for turn in range(turns):
            in_len = sp.input_len.sample(self.rng)
            out_len = sp.output_len.sample(self.rng)
            new_text = make_text(in_len, seed=self.rng.randrange(1 << 30))
            prompt = (history + "\n" + new_text) if history else new_text
            prompt_tokens = sys_tokens + _approx_tokens(prompt)
            yield Request(
                prompt=prompt,
                max_tokens=out_len,
                system=system,
                session_id=sid,
                tenant=tenant,
                prompt_tokens_hint=prompt_tokens,
                extra_headers={"x-session-id": sid, "x-tenant": tenant},
            )
            # Simulate the assistant reply joining the history so the next turn
            # reuses everything before it.
            history = prompt + "\n" + make_text(out_len, seed=self.rng.randrange(1 << 30))

    def stream(self, n: int) -> Iterator[Request]:
        """`n` requests, drawing whole sessions so multi-turn prefix reuse is
        preserved rather than interleaved randomly."""
        produced = 0
        while produced < n:
            for req in self.session():
                yield req
                produced += 1
                if produced >= n:
                    return


def _approx_tokens(text: str) -> int:
    return len(text.split())


# --- ready-made specs used across the repo -------------------------------
PRESETS: dict[str, WorkloadSpec] = {
    "chat": WorkloadSpec(
        name="chat",
        input_len=LengthSpec("lognormal", 512, sigma=0.7),
        output_len=LengthSpec("lognormal", 200, sigma=0.6),
    ),
    "short": WorkloadSpec(
        name="short",
        input_len=LengthSpec("fixed", 128),
        output_len=LengthSpec("fixed", 64),
    ),
    "balanced": WorkloadSpec(
        name="balanced",
        input_len=LengthSpec("fixed", 1024),
        output_len=LengthSpec("fixed", 256),
    ),
    "prefill_heavy": WorkloadSpec(
        name="prefill_heavy",
        input_len=LengthSpec("fixed", 8192),
        output_len=LengthSpec("fixed", 32),
    ),
    "decode_heavy": WorkloadSpec(
        name="decode_heavy",
        input_len=LengthSpec("fixed", 128),
        output_len=LengthSpec("fixed", 1024),
    ),
    "long_context": WorkloadSpec(
        name="long_context",
        input_len=LengthSpec("fixed", 32000),
        output_len=LengthSpec("fixed", 128),
    ),
    "rag_shared_prefix": WorkloadSpec(
        name="rag_shared_prefix",
        input_len=LengthSpec("fixed", 256),
        output_len=LengthSpec("fixed", 128),
        system_prompt_tokens=2048,
        n_prefix_variants=4,
    ),
    "multiturn_agent": WorkloadSpec(
        name="multiturn_agent",
        input_len=LengthSpec("fixed", 200),
        output_len=LengthSpec("fixed", 150),
        system_prompt_tokens=1024,
        n_prefix_variants=2,
        multi_turn=6,
    ),
}
