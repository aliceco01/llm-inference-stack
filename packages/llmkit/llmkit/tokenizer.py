"""Token counting with an explicit accuracy contract.

Benchmarks that estimate token counts and then report tokens/sec to three
significant figures are reporting their estimator, not the server. The order
of preference here is therefore:

1. Server-reported `usage` (exact, and what the provider bills on).
2. A real tokenizer for the model (exact, needs the vocab locally).
3. A character-ratio heuristic (approximate, and every summary derived from it
   is tagged `token_source=heuristic` so it can never be mistaken for measured).
"""

from __future__ import annotations

import functools
import os
import re
from typing import Protocol


class Counter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicCounter:
    """Character-count based estimate.

    The 3.6 chars/token default is the empirical mean for English prose on
    Llama-family BPE vocabularies. Code and non-Latin scripts diverge sharply
    (code runs denser, CJK runs far denser), so the ratio is configurable and
    the source is always reported.
    """

    name = "heuristic"
    exact = False

    def __init__(self, chars_per_token: float = 3.6) -> None:
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        # Whitespace-delimited words are a better anchor than raw chars for
        # prose; fall back to chars when there is little whitespace (code, CJK).
        n_chars = len(text)
        n_words = len(re.findall(r"\S+", text))
        if n_words and n_chars / max(n_words, 1) < 12:
            est = n_words * 1.3
        else:
            est = n_chars / self.chars_per_token
        return max(1, int(round(est)))


class HFCounter:
    """Exact counts from a Hugging Face tokenizer.json, when one is available."""

    exact = True

    def __init__(self, model_or_path: str) -> None:
        from tokenizers import Tokenizer  # type: ignore

        if os.path.exists(model_or_path):
            self._tk = Tokenizer.from_file(model_or_path)
        else:
            self._tk = Tokenizer.from_pretrained(model_or_path)
        self.name = f"hf:{model_or_path}"

    def count(self, text: str) -> int:
        return len(self._tk.encode(text, add_special_tokens=False).ids)


@functools.lru_cache(maxsize=8)
def get_counter(model: str | None = None) -> Counter:
    """Best available counter for `model`, degrading loudly rather than silently."""
    path = os.environ.get("LLMKIT_TOKENIZER") or model
    if path:
        try:
            return HFCounter(path)
        except Exception:
            pass
    return HeuristicCounter()


def token_source(counter: Counter) -> str:
    return "exact" if getattr(counter, "exact", False) else "heuristic"
