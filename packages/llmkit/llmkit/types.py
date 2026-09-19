"""Canonical data types shared by every project in this repo.

The whole stack is built around one rule: a measurement produced by the real
vLLM/SGLang client and a measurement produced by the simulator are the *same
type*, so every downstream analysis, plot and report works on both without
branching. That is what makes 15 projects one system instead of 15 scripts.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def now_ns() -> int:
    """Monotonic high-resolution clock.

    time.perf_counter_ns() is monotonic and not subject to NTP steps, which
    matters because a wall-clock adjustment mid-run can otherwise produce
    negative inter-token latencies.
    """
    return time.perf_counter_ns()


UNSET = -1  # sentinel for 'no timestamp'; 0 is a legal virtual-clock value

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


class Phase(str, Enum):
    """Which engine phase a scheduler step spent its budget on."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PREEMPTED_DROP = "preempted_drop"


@dataclass
class TokenEvent:
    """One streamed chunk, timestamped at the moment of arrival.

    `content_tokens` is the number of tokens carried by this chunk. It is
    usually 1, but servers batch multiple tokens into a single SSE frame under
    load, and treating every frame as one token silently inflates the measured
    token rate. We record the true count and normalise later.
    """

    t_ns: int
    content_tokens: int = 1
    text: str = ""


@dataclass
class RequestRecord:
    """The complete lifecycle of a single inference request.

    Every latency metric in this repo is derived from this record, so the
    fields here are deliberately raw timestamps rather than pre-computed
    latencies: derived metrics can be recomputed and audited, baked-in ones
    cannot.
    """

    request_id: str
    # --- lifecycle timestamps (monotonic ns) ---
    t_submit_ns: int = UNSET          # handed to the load generator
    t_send_ns: int = UNSET            # bytes actually on the wire
    t_first_chunk_ns: int = UNSET     # first SSE frame of any kind (may be role-only)
    t_first_token_ns: int = UNSET     # first frame carrying actual content
    t_last_token_ns: int = UNSET
    t_done_ns: int = UNSET

    # --- shape ---
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_prompt_tokens: int = 0  # prefix-cache hit, when the server reports it

    # --- outcome ---
    finish_reason: FinishReason = FinishReason.STOP
    error: str | None = None
    status_code: int | None = None

    # --- provenance ---
    replica: str | None = None     # which backend served it (gateway/proxy runs)
    model: str | None = None
    tenant: str | None = None
    session_id: str | None = None  # multi-turn grouping for prefix-cache tests

    token_events: list[TokenEvent] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived latency metrics. All in milliseconds.
    # ------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self.error is None and self.finish_reason in (
            FinishReason.STOP,
            FinishReason.LENGTH,
        )

    @property
    def queue_ms(self) -> float:
        """Time spent in the load generator before hitting the wire.

        Non-zero only in open-loop mode when the client itself is saturated.
        A rising value means the *benchmark* is the bottleneck, not the server,
        which is the single most common way load tests lie.
        """
        if self.t_send_ns < 0 or self.t_submit_ns < 0:
            return 0.0
        return (self.t_send_ns - self.t_submit_ns) / NS_PER_MS

    @property
    def ttft_ms(self) -> float:
        """Time to first *content* token, measured from send.

        Deliberately not measured from the first SSE frame: OpenAI-compatible
        servers emit a role-only delta first, and counting it as the first
        token under-reports TTFT by a full scheduler step.
        """
        if self.t_first_token_ns < 0 or self.t_send_ns < 0:
            return math.nan
        return (self.t_first_token_ns - self.t_send_ns) / NS_PER_MS

    @property
    def ttft_first_chunk_ms(self) -> float:
        """TTFT as naively measured from the first SSE frame. Kept so the
        difference between the two definitions can be shown, not hidden."""
        if self.t_first_chunk_ns < 0 or self.t_send_ns < 0:
            return math.nan
        return (self.t_first_chunk_ns - self.t_send_ns) / NS_PER_MS

    @property
    def e2e_ms(self) -> float:
        if self.t_done_ns < 0 or self.t_send_ns < 0:
            return math.nan
        return (self.t_done_ns - self.t_send_ns) / NS_PER_MS

    @property
    def itls_ms(self) -> list[float]:
        """Inter-token latencies: the gaps between successive content tokens.

        Chunks carrying k>1 tokens are amortised evenly across the gap rather
        than reported as one large ITL, otherwise server-side batching of SSE
        frames shows up as a fake latency spike.
        """
        evs = [e for e in self.token_events if e.content_tokens > 0]
        if len(evs) < 2:
            return []
        out: list[float] = []
        for prev, cur in zip(evs, evs[1:]):
            gap_ms = (cur.t_ns - prev.t_ns) / NS_PER_MS
            n = max(1, cur.content_tokens)
            out.extend([gap_ms / n] * n)
        return out

    @property
    def tpot_ms(self) -> float:
        """Time per output token: the mean decode-step cost for this request.

        TPOT is a single number per request; ITL is a distribution. Reporting
        only TPOT hides stalls caused by preemption and queue interference,
        which is exactly what a load curve is supposed to reveal.
        """
        if self.output_tokens < 2 or math.isnan(self.ttft_ms):
            return math.nan
        return (self.e2e_ms - self.ttft_ms) / (self.output_tokens - 1)

    @property
    def decode_tok_per_s(self) -> float:
        if self.output_tokens < 2 or self.t_last_token_ns < 0 or self.t_first_token_ns < 0:
            return math.nan
        span_s = (self.t_last_token_ns - self.t_first_token_ns) / NS_PER_S
        if span_s <= 0:
            return math.nan
        return (self.output_tokens - 1) / span_s

    def to_row(self) -> dict[str, Any]:
        """Flat, storage-friendly projection. Token events are summarised to
        keep result files small; raw events stay available in memory."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "replica": self.replica,
            "tenant": self.tenant,
            "session_id": self.session_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "ok": self.ok,
            "finish_reason": self.finish_reason.value,
            "status_code": self.status_code,
            "error": self.error,
            "queue_ms": self.queue_ms,
            "ttft_ms": self.ttft_ms,
            "ttft_first_chunk_ms": self.ttft_first_chunk_ms,
            "e2e_ms": self.e2e_ms,
            "tpot_ms": self.tpot_ms,
            "decode_tok_per_s": self.decode_tok_per_s,
            "t_send_ns": self.t_send_ns,
            "t_done_ns": self.t_done_ns,
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }


@dataclass
class SLO:
    """Service level objective used for goodput.

    Throughput without an SLO is meaningless: you can always raise tokens/sec
    by letting latency go to infinity. Goodput is the only throughput number
    that cannot be gamed this way.
    """

    ttft_ms: float = 2000.0
    p_itl_ms: float = 100.0      # per-token latency ceiling
    itl_percentile: float = 95.0  # applied at this percentile within a request
    e2e_ms: float | None = None

    def met_by(self, r: RequestRecord) -> bool:
        if not r.ok:
            return False
        if math.isnan(r.ttft_ms) or r.ttft_ms > self.ttft_ms:
            return False
        itls = r.itls_ms
        if itls:
            from .metrics import percentile

            if percentile(itls, self.itl_percentile) > self.p_itl_ms:
                return False
        if self.e2e_ms is not None and (math.isnan(r.e2e_ms) or r.e2e_ms > self.e2e_ms):
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
