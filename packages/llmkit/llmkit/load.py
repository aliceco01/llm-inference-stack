"""Load drivers: closed loop and open loop.

This distinction is the single most important thing in the benchmark suite.

Closed loop (fixed concurrency N): each worker sends, waits for the full
response, then sends again. In-flight count is capped at N by construction.
It measures the server at a known operating point and it can never overload
the server, because a slow server simply receives requests more slowly. This
is what almost every "we did 64 concurrent requests" benchmark does, and it
systematically hides queueing: latency rises smoothly and nothing ever breaks.

Open loop (fixed arrival rate lambda): requests are born on a Poisson schedule
regardless of whether earlier ones finished. In-flight count is unbounded. If
service rate falls below arrival rate the queue grows without limit and latency
diverges. This is how real traffic behaves, and it is the only mode that can
find the capacity cliff.

Real answer: report both. Closed loop gives clean per-point latency, open loop
gives the SLO-bounded capacity. The suite refuses to publish a capacity claim
from closed-loop data alone.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from .client import Request, StreamingClient
from .types import NS_PER_S, RequestRecord, now_ns

Sender = Callable[[Request, int], Awaitable[RequestRecord]]


@dataclass
class DriverResult:
    records: list[RequestRecord] = field(default_factory=list)
    started_ns: int = 0
    ended_ns: int = 0
    scheduled: int = 0
    completed: int = 0
    max_inflight: int = 0
    schedule_lag_ms_p95: float = 0.0  # how far behind the arrival schedule we ran

    @property
    def duration_s(self) -> float:
        return max((self.ended_ns - self.started_ns) / NS_PER_S, 1e-9)


def _sender_from_client(client: StreamingClient) -> Sender:
    async def _send(req: Request, submitted_ns: int) -> RequestRecord:
        return await client.send(req, submitted_ns=submitted_ns)

    return _send


class _InflightTracker:
    def __init__(self) -> None:
        self.cur = 0
        self.peak = 0

    def enter(self) -> None:
        self.cur += 1
        self.peak = max(self.peak, self.cur)

    def exit(self) -> None:
        self.cur -= 1


async def run_closed_loop(
    sender: Sender | StreamingClient,
    requests: Iterable[Request],
    *,
    concurrency: int,
    duration_s: float | None = None,
    max_requests: int | None = None,
    on_record: Callable[[RequestRecord], None] | None = None,
) -> DriverResult:
    """N workers pulling from a shared request iterator."""
    send = _sender_from_client(sender) if isinstance(sender, StreamingClient) else sender
    it: Iterator[Request] = iter(requests)
    lock = asyncio.Lock()
    res = DriverResult(started_ns=now_ns())
    deadline = res.started_ns + int((duration_s or 0) * NS_PER_S) if duration_s else None
    track = _InflightTracker()
    stop = asyncio.Event()

    async def worker() -> None:
        while not stop.is_set():
            if deadline and now_ns() >= deadline:
                return
            async with lock:
                if max_requests is not None and res.scheduled >= max_requests:
                    return
                try:
                    req = next(it)
                except StopIteration:
                    return
                res.scheduled += 1
            t = now_ns()
            track.enter()
            try:
                rec = await send(req, t)
            finally:
                track.exit()
            res.records.append(rec)
            res.completed += 1
            if on_record:
                on_record(rec)

    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    res.ended_ns = now_ns()
    res.max_inflight = track.peak
    return res


async def run_open_loop(
    sender: Sender | StreamingClient,
    requests: Iterable[Request],
    *,
    rps: float,
    duration_s: float,
    max_inflight: int = 20000,
    arrival: str = "poisson",
    seed: int = 7,
    drain_timeout_s: float = 120.0,
    on_record: Callable[[RequestRecord], None] | None = None,
) -> DriverResult:
    """Poisson (or uniform) arrivals at `rps`, independent of completions.

    The arrival schedule is absolute, computed from the run start, so a slow
    dispatch loop cannot silently stretch the intended rate. Lag against that
    schedule is measured and reported: if it grows, the *client* is the
    bottleneck and the run must be discarded rather than published.
    """
    send = _sender_from_client(sender) if isinstance(sender, StreamingClient) else sender
    rng = random.Random(seed)
    it: Iterator[Request] = iter(requests)
    res = DriverResult(started_ns=now_ns())
    track = _InflightTracker()
    tasks: set[asyncio.Task] = set()
    lags: list[float] = []
    sem = asyncio.Semaphore(max_inflight)

    async def fire(req: Request, submitted_ns: int) -> None:
        async with sem:
            track.enter()
            try:
                rec = await send(req, submitted_ns)
            finally:
                track.exit()
            res.records.append(rec)
            res.completed += 1
            if on_record:
                on_record(rec)

    t0 = res.started_ns
    end_ns = t0 + int(duration_s * NS_PER_S)
    next_arrival = float(t0)
    loop = asyncio.get_running_loop()

    while True:
        if next_arrival >= end_ns:
            break
        now = now_ns()
        wait_s = (next_arrival - now) / NS_PER_S
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        else:
            lags.append(-wait_s * 1000.0)
            # Yield so completions can run even when we are behind schedule.
            await asyncio.sleep(0)
        try:
            req = next(it)
        except StopIteration:
            break
        res.scheduled += 1
        t = now_ns()
        task = loop.create_task(fire(req, t))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

        gap_s = rng.expovariate(rps) if arrival == "poisson" else 1.0 / rps
        next_arrival += gap_s * NS_PER_S

    if tasks:
        await asyncio.wait(tasks, timeout=drain_timeout_s)
    res.ended_ns = now_ns()
    res.max_inflight = track.peak
    if lags:
        lags.sort()
        idx = max(0, math.ceil(0.95 * len(lags)) - 1)
        res.schedule_lag_ms_p95 = lags[idx]
    return res


@dataclass
class SweepPoint:
    """One point on a load curve."""

    concurrency: int | None = None
    rps: float | None = None
    duration_s: float = 30.0
    max_requests: int | None = None


def concurrency_ladder(
    start: int = 1, stop: int = 256, factor: float = 2.0, extra: Sequence[int] = ()
) -> list[int]:
    """Geometric ladder. Geometric rather than linear because latency-vs-load
    is roughly hyperbolic near saturation, so linear steps waste most samples
    in the flat region and miss the knee entirely."""
    out: list[int] = []
    c = float(start)
    while c <= stop:
        v = int(round(c))
        if v not in out:
            out.append(v)
        c *= factor
    for e in extra:
        if e not in out:
            out.append(e)
    return sorted(out)
