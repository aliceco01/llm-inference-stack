"""Aggregation from raw RequestRecords into a reportable summary.

Design rules enforced here, each of which exists because the opposite is a
common way benchmarks mislead:

1. Throughput is measured over a *window*, never as the sum of per-request
   rates. Summing per-request rates counts idle time twice and inflates the
   number by roughly the concurrency factor.
2. Warmup is excluded explicitly and the excluded count is reported, so a
   reader can see how much data was dropped and re-derive the number.
3. ITL is pooled across requests but *also* reported as a per-request p95
   distribution, because a pooled percentile lets a handful of long requests
   dominate and hides per-user stalls.
4. Little's law is checked. If achieved concurrency does not match
   throughput * mean latency, the harness is lying and the run is flagged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import NS_PER_S, SLO, RequestRecord


def percentile(values: Sequence[float], p: float, method: str = "nearest_rank") -> float:
    """Percentile of a sample.

    `nearest_rank` is the default because it is the convention for latency
    SLOs: the reported p99 is an value that actually occurred, never an
    interpolated value that no request ever experienced.
    """
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    p = min(max(p, 0.0), 100.0)
    if method == "nearest_rank":
        rank = math.ceil(p / 100.0 * len(xs))
        return xs[max(1, rank) - 1]
    # linear interpolation (numpy-compatible), for cross-checking
    idx = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return xs[int(idx)]
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def mean(values: Iterable[float]) -> float:
    xs = [v for v in values if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else math.nan


def stdev(values: Iterable[float]) -> float:
    xs = [v for v in values if not math.isnan(v)]
    if len(xs) < 2:
        return math.nan
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


@dataclass
class Dist:
    """Summary of one latency distribution, in milliseconds."""

    n: int = 0
    mean: float = math.nan
    std: float = math.nan
    p50: float = math.nan
    p90: float = math.nan
    p95: float = math.nan
    p99: float = math.nan
    max: float = math.nan

    @classmethod
    def of(cls, values: Sequence[float]) -> Dist:
        xs = [v for v in values if not math.isnan(v)]
        if not xs:
            return cls()
        return cls(
            n=len(xs),
            mean=mean(xs),
            std=stdev(xs),
            p50=percentile(xs, 50),
            p90=percentile(xs, 90),
            p95=percentile(xs, 95),
            p99=percentile(xs, 99),
            max=max(xs),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    """Everything needed to plot a load curve point and to audit it."""

    label: str = ""
    concurrency: int = 0
    target_rps: float | None = None

    n_total: int = 0
    n_ok: int = 0
    n_error: int = 0
    n_excluded_warmup: int = 0
    errors_by_kind: dict[str, int] = field(default_factory=dict)

    window_s: float = math.nan

    ttft: Dist = field(default_factory=Dist)
    itl: Dist = field(default_factory=Dist)           # pooled over all tokens
    itl_per_req_p95: Dist = field(default_factory=Dist)  # per-request p95, then summarised
    tpot: Dist = field(default_factory=Dist)
    e2e: Dist = field(default_factory=Dist)
    queue: Dist = field(default_factory=Dist)

    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_prompt_tokens: int = 0

    req_per_s: float = math.nan
    output_tok_per_s: float = math.nan
    total_tok_per_s: float = math.nan
    goodput_req_per_s: float = math.nan
    goodput_ratio: float = math.nan
    prefix_hit_rate: float = math.nan

    achieved_concurrency: float = math.nan
    littles_law_error: float = math.nan  # relative error; >0.15 means suspect
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def headline(self) -> str:
        return (
            f"{self.label:<22} c={self.concurrency:<4} "
            f"TTFT p50/p95 {self.ttft.p50:7.1f}/{self.ttft.p95:7.1f}ms  "
            f"ITL p95 {self.itl.p95:6.2f}ms  "
            f"out {self.output_tok_per_s:8.1f} tok/s  "
            f"goodput {self.goodput_ratio * 100:5.1f}%"
        )


def summarize(
    records: Sequence[RequestRecord],
    *,
    label: str = "",
    concurrency: int = 0,
    target_rps: float | None = None,
    slo: SLO | None = None,
    warmup_s: float = 0.0,
    warmup_requests: int = 0,
    meta: dict[str, Any] | None = None,
) -> RunSummary:
    """Collapse records into a single load-curve point.

    The measurement window is the span from the first send to the last
    completion *among retained requests*. Using wall time of the whole process
    instead would fold process startup and teardown into the throughput number.
    """
    slo = slo or SLO()
    s = RunSummary(label=label, concurrency=concurrency, target_rps=target_rps,
                   meta=dict(meta or {}))
    if not records:
        s.warnings.append("no records")
        return s

    ordered = sorted(records, key=lambda r: r.t_send_ns if r.t_send_ns >= 0 else 0)
    s.n_total = len(ordered)

    retained = list(ordered)
    if warmup_requests > 0:
        s.n_excluded_warmup += min(warmup_requests, len(retained))
        retained = retained[warmup_requests:]
    if warmup_s > 0 and retained:
        t0 = retained[0].t_send_ns
        cutoff = t0 + int(warmup_s * NS_PER_S)
        before = len(retained)
        retained = [r for r in retained if r.t_send_ns >= cutoff]
        s.n_excluded_warmup += before - len(retained)

    if not retained:
        s.warnings.append("all requests excluded by warmup filter")
        return s

    ok = [r for r in retained if r.ok]
    bad = [r for r in retained if not r.ok]
    s.n_ok, s.n_error = len(ok), len(bad)
    for r in bad:
        kind = r.error or r.finish_reason.value
        # collapse to a short kind so the table stays readable
        kind = kind.split(":")[0][:48]
        s.errors_by_kind[kind] = s.errors_by_kind.get(kind, 0) + 1

    # --- measurement window -------------------------------------------------
    t_start = min(r.t_send_ns for r in retained)
    t_end = max((r.t_done_ns for r in retained if r.t_done_ns >= 0), default=t_start)
    s.window_s = max((t_end - t_start) / NS_PER_S, 1e-9)

    # --- latency distributions (successful requests only) -------------------
    s.ttft = Dist.of([r.ttft_ms for r in ok])
    s.e2e = Dist.of([r.e2e_ms for r in ok])
    s.tpot = Dist.of([r.tpot_ms for r in ok])
    s.queue = Dist.of([r.queue_ms for r in retained])

    pooled_itl: list[float] = []
    per_req_p95: list[float] = []
    for r in ok:
        itls = r.itls_ms
        if itls:
            pooled_itl.extend(itls)
            per_req_p95.append(percentile(itls, 95))
    s.itl = Dist.of(pooled_itl)
    s.itl_per_req_p95 = Dist.of(per_req_p95)

    # --- token accounting ---------------------------------------------------
    s.prompt_tokens = sum(r.prompt_tokens for r in ok)
    s.output_tokens = sum(r.output_tokens for r in ok)
    s.cached_prompt_tokens = sum(r.cached_prompt_tokens for r in ok)
    if s.prompt_tokens > 0:
        s.prefix_hit_rate = s.cached_prompt_tokens / s.prompt_tokens

    # --- throughput over the window ----------------------------------------
    s.req_per_s = len(ok) / s.window_s
    s.output_tok_per_s = s.output_tokens / s.window_s
    s.total_tok_per_s = (s.output_tokens + s.prompt_tokens) / s.window_s

    good = [r for r in ok if slo.met_by(r)]
    s.goodput_req_per_s = len(good) / s.window_s
    s.goodput_ratio = len(good) / len(retained) if retained else math.nan

    # --- Little's law sanity check -----------------------------------------
    # N = X * R. Achieved concurrency is computed by integrating the number of
    # in-flight requests over the window, which is independent of the formula,
    # so agreement is a genuine cross-check of the harness.
    s.achieved_concurrency = _integrate_concurrency(retained, t_start, t_end)
    predicted = s.req_per_s * (mean([r.e2e_ms for r in ok]) / 1000.0)
    if not math.isnan(predicted) and predicted > 0 and s.achieved_concurrency > 0:
        s.littles_law_error = abs(s.achieved_concurrency - predicted) / predicted
        if s.littles_law_error > 0.15:
            s.warnings.append(
                f"Little's law mismatch: measured N={s.achieved_concurrency:.2f} "
                f"vs X*R={predicted:.2f} (err {s.littles_law_error*100:.0f}%). "
                "Check for client-side queueing or an unclosed measurement window."
            )

    if s.queue.p95 > 50:
        s.warnings.append(
            f"client-side queue p95 ={s.queue.p95:.0f}ms: the load generator is "
            "saturated, so server latency is being over-reported"
        )
    if s.n_error:
        s.warnings.append(f"{s.n_error}/{len(retained)} requests failed")
    return s


def _integrate_concurrency(records: Sequence[RequestRecord], t_start: int, t_end: int) -> float:
    """Time-weighted mean number of in-flight requests over [t_start, t_end]."""
    events: list[tuple[int, int]] = []
    for r in records:
        if r.t_send_ns < 0:
            continue
        done = r.t_done_ns if r.t_done_ns >= 0 else t_end
        events.append((r.t_send_ns, +1))
        events.append((done, -1))
    if not events:
        return math.nan
    events.sort()
    span = t_end - t_start
    if span <= 0:
        return math.nan
    area = 0
    cur = 0
    prev_t = t_start
    for t, delta in events:
        t_clamped = min(max(t, t_start), t_end)
        area += cur * (t_clamped - prev_t)
        prev_t = t_clamped
        cur += delta
    area += cur * (t_end - prev_t)
    return area / span


def sweep_table(summaries: Sequence[RunSummary]) -> str:
    """Render a concurrency sweep as a fixed-width table for the terminal."""
    hdr = (
        f"{'label':<20}{'conc':>5}{'RPS':>8}{'ok':>6}{'err':>5}"
        f"{'TTFT p50':>10}{'TTFT p95':>10}{'TTFT p99':>10}"
        f"{'ITL p50':>9}{'ITL p95':>9}{'out tok/s':>11}{'goodput':>9}"
    )
    lines = [hdr, "-" * len(hdr)]
    for s in summaries:
        lines.append(
            f"{s.label[:19]:<20}{s.concurrency:>5}{s.req_per_s:>8.2f}{s.n_ok:>6}{s.n_error:>5}"
            f"{s.ttft.p50:>10.1f}{s.ttft.p95:>10.1f}{s.ttft.p99:>10.1f}"
            f"{s.itl.p50:>9.2f}{s.itl.p95:>9.2f}{s.output_tok_per_s:>11.1f}"
            f"{s.goodput_ratio * 100:>8.1f}%"
        )
    return "\n".join(lines)


def find_knee(summaries: Sequence[RunSummary], slo: SLO) -> RunSummary | None:
    """Highest-concurrency point that still meets the SLO for >=95% of requests.

    This is the number that actually matters for capacity planning: peak
    throughput past the knee is throughput you cannot sell.
    """
    passing = [s for s in sorted(summaries, key=lambda x: x.concurrency)
               if not math.isnan(s.goodput_ratio) and s.goodput_ratio >= 0.95]
    return passing[-1] if passing else None
