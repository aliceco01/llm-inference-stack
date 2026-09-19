"""Minimal Prometheus exposition parsing.

Shared by the KV monitor (03), the acceptance tracker (06), the autoscaler (11),
the cost dashboard (12) and the chaos suite (14). Deliberately dependency-free:
pulling in a full Prometheus client to read five gauges off one endpoint is not
a trade worth making, and the exposition format is stable and simple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+|NaN)$"
)
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, float]:
    """Flatten exposition format, summing series that share a metric name.

    Summing across label sets is the right default here: a server may report
    one series per model, and the operator question ("is this replica out of KV
    cache") is about the replica as a whole. Use `parse_labeled` when the label
    values matter.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        val = m.group("value")
        if val == "NaN":
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        name = m.group("name")
        out[name] = out.get(name, 0.0) + v
    return out


def parse_labeled(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Every sample as (name, labels, value), preserving label sets."""
    out: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        val = m.group("value")
        if val == "NaN":
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out.append((m.group("name"), labels, v))
    return out


def histogram_quantile(text: str, metric: str, q: float) -> float:
    """Approximate quantile from a Prometheus histogram's buckets.

    Linear interpolation within the bucket that contains the target rank, which
    is what Prometheus itself does. Accuracy is bounded by bucket width, so a
    p99 read off coarse buckets is an estimate: prefer the client-side
    measurement from project 02 when precision matters.
    """
    buckets: list[tuple[float, float]] = []
    total = 0.0
    for name, labels, v in parse_labeled(text):
        if name == f"{metric}_bucket":
            le = labels.get("le")
            if le is None:
                continue
            buckets.append((float("inf") if le in ("+Inf", "Inf") else float(le), v))
        elif name == f"{metric}_count":
            total += v
    if not buckets or total <= 0:
        return float("nan")
    buckets.sort()
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    return buckets[-1][0]


async def scrape(url: str, *, timeout: float = 5.0,
                 client: httpx.AsyncClient | None = None) -> dict[str, float]:
    """Fetch and parse /metrics. Returns {} on any failure, never raises."""
    own = client is None
    c = client or httpx.AsyncClient(timeout=timeout)
    try:
        r = await c.get(url.rstrip("/") + "/metrics", timeout=timeout)
        r.raise_for_status()
        return parse_prometheus(r.text)
    except Exception:
        return {}
    finally:
        if own:
            await c.aclose()


def scrape_sync(url: str, *, timeout: float = 5.0) -> dict[str, float]:
    try:
        r = httpx.get(url.rstrip("/") + "/metrics", timeout=timeout)
        r.raise_for_status()
        return parse_prometheus(r.text)
    except Exception:
        return {}


@dataclass
class CounterRate:
    """Per-second rate of a monotonic counter, with reset detection.

    A counter that goes backwards means the process restarted. Treating that as
    a large negative rate produces nonsense; treating it as a reset and
    restarting the window is correct and is what Prometheus does.
    """

    name: str
    _last_value: float | None = None
    _last_t: float | None = None
    _rate: float = 0.0
    restarts: int = 0

    def update(self, value: float, t: float) -> float:
        if self._last_value is None or self._last_t is None:
            self._last_value, self._last_t = value, t
            return 0.0
        dt = t - self._last_t
        if dt <= 0:
            return self._rate
        if value < self._last_value:
            self.restarts += 1
            self._last_value, self._last_t = value, t
            return self._rate
        self._rate = (value - self._last_value) / dt
        self._last_value, self._last_t = value, t
        return self._rate

    @property
    def rate(self) -> float:
        return self._rate
