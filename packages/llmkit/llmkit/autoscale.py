"""Autoscaling policies for inference, and a simulator to evaluate them.

Why GPU utilisation is the wrong signal
---------------------------------------
A saturated inference server reports ~100% GPU utilisation whether it is
serving 10 requests or 1000. `nvidia-smi` utilisation measures "was a kernel
resident in the last sampling window", not "how much work is queued". Under
continuous batching the GPU is essentially always busy, so utilisation
saturates long before the server does and carries no information about unmet
demand.

Queue depth does. `vllm:num_requests_waiting` is a direct count of work that
has arrived and cannot be served yet, and it starts rising at exactly the
moment capacity becomes insufficient.

The complication: cold start
-----------------------------
A replica takes minutes to become useful (pull image, load weights, profile the
KV cache, capture CUDA graphs). A reactive policy scales up when the queue
builds and the new capacity lands several minutes later, often after the spike
has passed. So the policy has to either keep headroom, predict, or accept the
violation, and the point of the simulator here is to make that choice with
numbers instead of intuition.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Signal = Literal["queue_depth", "gpu_util", "running_requests", "concurrency_per_replica",
                 "ttft_p95", "queue_plus_predictive"]


@dataclass
class ReplicaState:
    """One replica's lifecycle. Cold start is explicit because it dominates."""

    replica_id: int
    state: str = "starting"        # starting | ready | draining | gone
    ready_at_s: float = 0.0
    drain_until_s: float = 0.0
    inflight: int = 0

    def is_serving(self, t: float) -> bool:
        return self.state == "ready" and t >= self.ready_at_s


@dataclass
class ScalerConfig:
    signal: Signal = "queue_depth"
    target: float = 8.0            # desired value of the signal per replica
    min_replicas: int = 1
    max_replicas: int = 20

    # Cold start, split into its real components so the model is honest about
    # which part you can actually optimise.
    image_pull_s: float = 20.0     # 0 if pre-pulled onto the node
    weight_load_s: float = 90.0    # dominated by storage bandwidth and size
    warmup_s: float = 25.0         # profiling + CUDA graph capture

    # Flapping control
    scale_up_cooldown_s: float = 30.0
    scale_down_cooldown_s: float = 300.0
    scale_down_stabilization_s: float = 300.0   # must stay low this long
    max_scale_up_step: int = 4     # replicas added per decision
    max_scale_down_step: int = 1

    # Drain
    drain_s: float = 45.0

    # Headroom / predictive
    headroom_factor: float = 1.0   # provision this multiple of demand
    predictive_horizon_s: float = 0.0   # >0 enables slope extrapolation

    @property
    def cold_start_s(self) -> float:
        return self.image_pull_s + self.weight_load_s + self.warmup_s


def desired_replicas(signal_value: float, cfg: ScalerConfig,
                     current_ready: int) -> int:
    """KEDA/HPA arithmetic: ceil(current * metric / target).

    KEDA's Prometheus scaler compares an aggregate query against a threshold
    that is expressed *per replica*, so the ratio form is what actually runs in
    production and is what is modelled here.
    """
    # The arithmetic is identical for every signal, which is itself the point:
    # the ratio form is only meaningful when the signal scales with unmet
    # demand. Queue depth does. GPU utilisation does not, so the same formula
    # that works for one silently misbehaves for the other.
    ratio = signal_value / max(cfg.target, 1e-9)
    want = math.ceil(max(current_ready, 1) * ratio * cfg.headroom_factor)
    return max(cfg.min_replicas, min(cfg.max_replicas, want))


@dataclass
class ScaleEvent:
    t: float
    action: str
    from_n: int
    to_n: int
    signal: float
    reason: str = ""


@dataclass
class AutoscaleResult:
    events: list[ScaleEvent] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    offered_rps: list[float] = field(default_factory=list)
    capacity_rps: list[float] = field(default_factory=list)
    queue: list[float] = field(default_factory=list)
    ready: list[int] = field(default_factory=list)
    starting: list[int] = field(default_factory=list)
    served: float = 0.0
    dropped: float = 0.0
    slo_violation_s: float = 0.0
    replica_seconds: float = 0.0
    wasted_replica_seconds: float = 0.0   # capacity paid for but not usable

    def summary(self) -> dict[str, float]:
        total = self.served + self.dropped
        gpu_hours = self.replica_seconds / 3600.0
        return {
            "served_requests": round(self.served),
            "dropped_requests": round(self.dropped),
            "drop_rate_pct": round(self.dropped / max(total, 1) * 100, 2),
            "slo_violation_s": round(self.slo_violation_s, 1),
            "slo_violation_pct": round(
                self.slo_violation_s / max(self.t[-1] if self.t else 1, 1) * 100, 1),
            "gpu_hours": round(gpu_hours, 3),
            "wasted_gpu_hours": round(self.wasted_replica_seconds / 3600.0, 3),
            "scale_events": len(self.events),
            "peak_replicas": max(self.ready) if self.ready else 0,
        }


def simulate(
    cfg: ScalerConfig,
    arrival_rps: Callable[[float], float],
    *,
    duration_s: float = 3600.0,
    dt_s: float = 5.0,
    capacity_per_replica_rps: float = 10.0,
    queue_slo: float = 20.0,
    max_queue: float = 5000.0,
    seed: int = 0,
) -> AutoscaleResult:
    """Fluid-flow simulation of an autoscaled pool.

    Requests are modelled as a rate rather than individually: the question here
    is capacity dynamics over minutes, and per-request detail would slow it
    down without changing the answer. Queue depth, cold start, cooldowns and
    drain are all modelled explicitly because those are what the policy
    actually interacts with.
    """
    rng = random.Random(seed)
    res = AutoscaleResult()
    replicas: list[ReplicaState] = []
    next_id = 0
    for _ in range(cfg.min_replicas):
        replicas.append(ReplicaState(next_id, "ready", ready_at_s=0.0)); next_id += 1

    queue = 0.0
    last_up = -1e9
    last_down = -1e9
    low_since: float | None = None
    signal_hist: list[tuple[float, float]] = []

    t = 0.0
    while t < duration_s:
        ready = [r for r in replicas if r.is_serving(t)]
        starting = [r for r in replicas if r.state == "starting"]
        n_ready = len(ready)

        offered = max(arrival_rps(t), 0.0)
        capacity = n_ready * capacity_per_replica_rps

        # --- queue dynamics ------------------------------------------------
        arrived = offered * dt_s
        servable = capacity * dt_s
        queue += arrived
        done = min(queue, servable)
        queue -= done
        res.served += done
        if queue > max_queue:
            res.dropped += queue - max_queue
            queue = max_queue

        if queue / max(n_ready, 1) > queue_slo:
            res.slo_violation_s += dt_s

        res.replica_seconds += (len(ready) + len(starting)) * dt_s
        res.wasted_replica_seconds += len(starting) * dt_s

        # --- signal --------------------------------------------------------
        if cfg.signal == "gpu_util":
            # The pathology, modelled faithfully: utilisation pins at ~1.0 as
            # soon as there is any sustained work, and stays there no matter
            # how deep the queue gets.
            sig = 1.0 if capacity > 0 and offered > capacity * 0.35 else (
                offered / max(capacity, 1e-9))
            sig = min(sig, 1.0) * 100.0
        elif cfg.signal == "running_requests":
            sig = min(queue + done, n_ready * 64) / max(n_ready, 1)
        elif cfg.signal == "ttft_p95":
            # Latency lags the cause it is meant to detect: it only rises after
            # the queue is already deep.
            sig = 50.0 + (queue / max(n_ready, 1)) * 120.0
        else:
            sig = queue / max(n_ready, 1)

        signal_hist.append((t, sig))
        signal_hist = [(tt, ss) for tt, ss in signal_hist if tt >= t - 120.0]

        effective_sig = sig
        if cfg.predictive_horizon_s > 0 and len(signal_hist) >= 4:
            # Linear extrapolation of the signal's recent slope. Crude, but it
            # is the mechanism that buys back part of the cold-start delay, and
            # its failure mode (overshoot on a spike that stops) is visible in
            # wasted_gpu_hours.
            t0, s0 = signal_hist[0]
            t1, s1 = signal_hist[-1]
            slope = (s1 - s0) / max(t1 - t0, 1e-9)
            effective_sig = max(sig, sig + slope * cfg.predictive_horizon_s)

        want = desired_replicas(effective_sig, cfg, n_ready)
        n_total = n_ready + len(starting)

        # --- scale up ------------------------------------------------------
        if want > n_total and (t - last_up) >= cfg.scale_up_cooldown_s:
            add = min(want - n_total, cfg.max_scale_up_step)
            for _ in range(add):
                jitter = rng.uniform(0.9, 1.15)
                replicas.append(ReplicaState(
                    next_id, "starting",
                    ready_at_s=t + cfg.cold_start_s * jitter))
                next_id += 1
            last_up = t
            low_since = None
            res.events.append(ScaleEvent(t, "up", n_total, n_total + add, sig,
                                         f"signal {sig:.1f} > target {cfg.target}"))
        # --- scale down ----------------------------------------------------
        elif want < n_ready:
            if low_since is None:
                low_since = t
            elif ((t - low_since) >= cfg.scale_down_stabilization_s
                  and (t - last_down) >= cfg.scale_down_cooldown_s):
                remove = min(n_ready - want, cfg.max_scale_down_step)
                for r in ready[-remove:]:
                    r.state = "draining"
                    r.drain_until_s = t + cfg.drain_s
                last_down = t
                low_since = None
                res.events.append(ScaleEvent(t, "down", n_ready, n_ready - remove,
                                             sig, "sustained low signal"))
        else:
            low_since = None

        # --- lifecycle transitions ----------------------------------------
        for r in replicas:
            if r.state == "starting" and t >= r.ready_at_s:
                r.state = "ready"
            elif r.state == "draining" and t >= r.drain_until_s:
                r.state = "gone"
        replicas = [r for r in replicas if r.state != "gone"]

        res.t.append(t)
        res.offered_rps.append(offered)
        res.capacity_rps.append(capacity)
        res.queue.append(queue)
        res.ready.append(n_ready)
        res.starting.append(len(starting))
        t += dt_s

    return res


# ---------------------------------------------------------------------------
# Traffic shapes
# ---------------------------------------------------------------------------
def spike(base: float = 5.0, peak: float = 60.0, at_s: float = 900.0,
          width_s: float = 300.0) -> Callable[[float], float]:
    """Sharp spike. The shape cold start handles worst."""
    def f(t: float) -> float:
        return base + (peak - base) * math.exp(-((t - at_s) ** 2) / (2 * (width_s / 3) ** 2))
    return f


def step(base: float = 5.0, peak: float = 40.0, at_s: float = 600.0) -> Callable[[float], float]:
    def f(t: float) -> float:
        return peak if t >= at_s else base
    return f


def diurnal(low: float = 4.0, high: float = 40.0, period_s: float = 3600.0
            ) -> Callable[[float], float]:
    def f(t: float) -> float:
        return low + (high - low) * 0.5 * (1 - math.cos(2 * math.pi * t / period_s))
    return f


def sawtooth_bursts(base: float = 4.0, peak: float = 45.0, period_s: float = 600.0
                    ) -> Callable[[float], float]:
    """Repeated bursts: the shape that exposes flapping."""
    def f(t: float) -> float:
        phase = (t % period_s) / period_s
        return peak if phase < 0.3 else base
    return f


TRAFFIC = {
    "spike": spike, "step": step, "diurnal": diurnal,
    "bursts": sawtooth_bursts,
}
