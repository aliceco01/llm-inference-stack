"""Gateway primitives: rate limits, retry budgets, breakers, degradation chains.

Provider outages are guaranteed; user-facing errors are optional. The gap
between those two statements is this module.

Three things here are less obvious than they look:

**Retry budgets, not retry counts.** Per-request retry limits are the standard
approach and they amplify load exactly when the fleet can least afford it: when
a backend degrades, every request retries, and offered load multiplies by the
retry count at the worst possible moment. A budget caps retries as a *fraction
of total traffic* (Google SRE's pattern), so retries stay helpful for isolated
failures and cannot turn a partial outage into a total one.

**Hedging must cancel the loser.** Sending a duplicate request when the first
has not produced a token by some deadline is a very effective tail-latency tool,
but only if the losing request is cancelled. Otherwise hedging at p95 adds a
permanent 5% load increase, which raises p95, which triggers more hedging.

**Token-bucket limits need two dimensions.** LLM traffic is not well described
by requests per minute: one request can be 200 tokens or 200,000. Limiting RPM
alone lets a single tenant with long prompts saturate a fleet while staying
inside its quota. Limit both requests and tokens.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
@dataclass
class TokenBucket:
    """Classic token bucket. `capacity` sets the burst, `rate` the sustained
    allowance."""

    rate: float                 # units per second
    capacity: float             # burst size
    tokens: float = 0.0
    last: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            self.tokens = self.capacity

    def _refill(self, now: float) -> None:
        dt = max(now - self.last, 0.0)
        self.tokens = min(self.capacity, self.tokens + dt * self.rate)
        self.last = now

    def try_consume(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def retry_after_s(self, n: float = 1.0) -> float:
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= n:
            return 0.0
        return (n - self.tokens) / max(self.rate, 1e-9)


@dataclass
class TenantLimits:
    """Per-tenant quota across both dimensions that matter."""

    tenant: str
    rpm: float = 600.0
    tpm: float = 600_000.0        # tokens per minute (prompt + completion)
    burst_requests: float = 60.0
    burst_tokens: float = 60_000.0
    max_concurrent: int = 64
    usd_budget_per_hour: float | None = None

    _req: TokenBucket = field(init=False, repr=False)
    _tok: TokenBucket = field(init=False, repr=False)
    inflight: int = 0
    spent_usd: float = 0.0
    _budget_window_start: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._req = TokenBucket(self.rpm / 60.0, self.burst_requests)
        self._tok = TokenBucket(self.tpm / 60.0, self.burst_tokens)

    def check(self, est_tokens: int) -> tuple[bool, str, float]:
        """(allowed, reason, retry_after_seconds)."""
        if self.inflight >= self.max_concurrent:
            return False, "max_concurrent", 1.0
        if self.usd_budget_per_hour is not None:
            now = time.monotonic()
            if now - self._budget_window_start >= 3600:
                self.spent_usd = 0.0
                self._budget_window_start = now
            if self.spent_usd >= self.usd_budget_per_hour:
                return False, "budget_exhausted", 3600 - (now - self._budget_window_start)
        if not self._req.try_consume(1.0):
            return False, "rpm", self._req.retry_after_s(1.0)
        if not self._tok.try_consume(est_tokens):
            # The request bucket was already debited. Refund it so a
            # token-limit rejection does not also consume request quota, which
            # would make a tenant's effective RPM depend on its prompt sizes.
            self._req.tokens = min(self._req.capacity, self._req.tokens + 1.0)
            return False, "tpm", self._tok.retry_after_s(est_tokens)
        return True, "", 0.0

    def settle(self, actual_tokens: int, est_tokens: int, usd: float = 0.0) -> None:
        """Reconcile the estimate against reality after the response.

        Completion length is unknown at admission, so the estimate is always
        wrong. Settling keeps the bucket honest over time instead of letting
        tenants with long outputs systematically under-pay.
        """
        delta = actual_tokens - est_tokens
        if delta > 0:
            self._tok.tokens = max(self._tok.tokens - delta, -self._tok.capacity)
        else:
            self._tok.tokens = min(self._tok.capacity, self._tok.tokens - delta)
        self.spent_usd += usd


# ---------------------------------------------------------------------------
# Retry budget
# ---------------------------------------------------------------------------
@dataclass
class RetryBudget:
    """Caps retries as a fraction of total traffic over a sliding window.

    This is the mechanism that prevents retry amplification. With a plain
    per-request retry count, a backend that starts failing causes every request
    to retry, tripling offered load precisely when capacity is already
    insufficient. A budget makes retries a scarce shared resource.
    """

    ratio: float = 0.1            # retries allowed per unit of normal traffic
    window_s: float = 10.0
    min_per_s: float = 1.0        # always allow a trickle, for isolated failures

    _requests: deque[float] = field(default_factory=deque, repr=False)
    _retries: deque[float] = field(default_factory=deque, repr=False)
    denied: int = 0

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        while self._retries and self._retries[0] < cutoff:
            self._retries.popleft()

    def record_request(self) -> None:
        now = time.monotonic()
        self._requests.append(now)
        self._trim(now)

    def try_retry(self) -> bool:
        now = time.monotonic()
        self._trim(now)
        allowed = max(len(self._requests) * self.ratio,
                      self.min_per_s * self.window_s)
        if len(self._retries) < allowed:
            self._retries.append(now)
            return True
        self.denied += 1
        return False

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        self._trim(now)
        return {"requests_in_window": len(self._requests),
                "retries_in_window": len(self._retries),
                "retry_ratio": round(len(self._retries) / max(len(self._requests), 1), 4),
                "denied": self.denied}


def backoff_delay(attempt: int, *, base_s: float = 0.05, cap_s: float = 2.0,
                  jitter: float = 1.0) -> float:
    """Exponential backoff with full jitter.

    Full jitter (uniform in [0, delay]) rather than fixed backoff, because
    synchronised clients retrying at the same computed instant reproduce the
    thundering herd the backoff was meant to prevent.
    """
    import random
    delay = min(cap_s, base_s * (2 ** attempt))
    return random.uniform(0, delay) if jitter else delay


# ---------------------------------------------------------------------------
# Degradation chain
# ---------------------------------------------------------------------------
Tier = Literal["primary", "secondary", "cheaper_model", "cached", "refuse"]


@dataclass
class Backend:
    """One place a request can go."""

    name: str
    base_url: str
    model: str
    tier: Tier = "primary"
    api_key_env: str | None = None
    weight: float = 1.0
    usd_per_m_prompt: float = 0.0
    usd_per_m_output: float = 0.0
    ttft_slo_ms: float = 2000.0
    max_concurrent: int = 512
    self_hosted: bool = True

    # live state
    inflight: int = 0
    total: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    # Consecutive failed health probes, tracked separately from request
    # failures. A probe failure is weaker evidence than a failed request, so it
    # needs more of them to trip the breaker, but it must count for something:
    # otherwise an unreachable backend stays advertised until a user finds it.
    probe_failures: int = 0
    open_until: float = 0.0
    ewma_ttft_ms: float = 0.0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.open_until and self.inflight < self.max_concurrent

    def observe_success(self, ttft_ms: float | None = None) -> None:
        self.consecutive_errors = 0
        if ttft_ms is not None:
            self.ewma_ttft_ms = (ttft_ms if not self.ewma_ttft_ms
                                 else 0.2 * ttft_ms + 0.8 * self.ewma_ttft_ms)

    def observe_error(self, *, threshold: int = 5, open_s: float = 15.0) -> None:
        self.errors += 1
        self.consecutive_errors += 1
        if self.consecutive_errors >= threshold:
            self.open_until = time.monotonic() + open_s

    def estimate_usd(self, prompt_tokens: int, output_tokens: int) -> float:
        return (prompt_tokens * self.usd_per_m_prompt
                + output_tokens * self.usd_per_m_output) / 1e6

    def meeting_slo(self) -> bool:
        return not self.ewma_ttft_ms or self.ewma_ttft_ms <= self.ttft_slo_ms


TIER_ORDER: list[Tier] = ["primary", "secondary", "cheaper_model", "cached", "refuse"]


@dataclass
class DegradationChain:
    """Ordered fallback tiers.

    The important design point is that the chain ends in something that is not
    an error. A gateway whose last resort is a 503 has not degraded, it has
    failed. A cached or canned response is worse than a fresh generation and
    much better than nothing, and deciding that in advance is what makes an
    outage survivable.
    """

    backends: list[Backend] = field(default_factory=list)
    allow_cached_response: bool = True
    cached_response_max_age_s: float = 3600.0

    def candidates(self, *, exclude: Sequence[str] = (),
                   require_slo: bool = False) -> list[Backend]:
        out: list[Backend] = []
        for tier in TIER_ORDER:
            tier_backends = [
                b for b in self.backends
                if b.tier == tier and b.name not in exclude and b.available
            ]
            if require_slo:
                meeting = [b for b in tier_backends if b.meeting_slo()]
                tier_backends = meeting or tier_backends
            tier_backends.sort(key=lambda b: (b.inflight / max(b.weight, 1e-9),
                                              b.ewma_ttft_ms))
            out.extend(tier_backends)
        return out

    def describe(self) -> str:
        lines = []
        for tier in TIER_ORDER:
            bs = [b for b in self.backends if b.tier == tier]
            if not bs:
                continue
            status = ", ".join(
                f"{b.name}{'' if b.available else ' (BREAKER OPEN)'}" for b in bs)
            lines.append(f"  {tier:<14} {status}")
        if self.allow_cached_response:
            lines.append(f"  {'cached':<14} responses up to "
                         f"{self.cached_response_max_age_s:.0f}s old")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hedging
# ---------------------------------------------------------------------------
@dataclass
class HedgeConfig:
    """Send a duplicate request when the first is slow.

    `delay_ms` should be set near the p95 TTFT of the primary, not the mean: at
    p95 the extra load is ~5% and it targets exactly the requests that were
    going to be slow. At the mean it doubles fleet load to fix nothing.
    """

    enabled: bool = False
    delay_ms: float = 500.0
    max_hedges: int = 1
    # Hedging is only safe for idempotent work. Generation is idempotent in the
    # sense that a duplicate costs money but does not corrupt state; anything
    # with tool side effects is not.
    only_if_no_tokens_yet: bool = True


async def hedged_call(primary, secondary, *, delay_s: float):
    """Race two coroutines, cancelling the loser.

    Returns (result, winner_index). Cancelling the loser is not optional: an
    uncancelled hedge is a permanent load increase proportional to the hedge
    rate.
    """
    t_primary = asyncio.create_task(primary())
    done, _ = await asyncio.wait({t_primary}, timeout=delay_s)
    if done:
        return t_primary.result(), 0

    t_secondary = asyncio.create_task(secondary())
    done, pending = await asyncio.wait(
        {t_primary, t_secondary}, return_when=asyncio.FIRST_COMPLETED)
    winner = next(iter(done))
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return winner.result(), (0 if winner is t_primary else 1)
