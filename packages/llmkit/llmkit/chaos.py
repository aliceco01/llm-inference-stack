"""Fault injection and SLO burn accounting.

Chaos engineering for inference differs from the general case in one important
way: the characteristic failure is not a crash, it is **gray failure**. A
replica whose GPU is thermally throttled, or whose KV cache is thrashing, stays
up, answers `/health` in single-digit milliseconds, and serves tokens at a
third of its normal rate. Every liveness check passes while users suffer, and
load balancers keep sending it traffic because nothing looks wrong.

So the suite injects slowness as a first-class fault, not just kills, and it
measures recovery rather than only failure.

Error budget accounting follows the standard SRE model. Burn rate is the
multiple of the budget-consumption rate that would exactly exhaust the budget
over the SLO window:

    burn_rate = observed_failure_ratio / (1 - slo_target)

A burn rate of 1 exhausts the budget precisely at the end of the window; 14.4
exhausts a 30-day budget in about 2 days, which is the conventional
page-immediately threshold.
"""

from __future__ import annotations

import asyncio
import math
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

FaultKind = Literal["throttle", "kill", "pause", "errors", "latency",
                    "kv_pressure", "spike", "partition"]


# ---------------------------------------------------------------------------
# SLO accounting
# ---------------------------------------------------------------------------
@dataclass
class SLOTarget:
    """An availability/latency objective and its error budget."""

    name: str = "ttft_p95"
    target: float = 0.99          # fraction of requests that must succeed
    window_days: float = 30.0

    @property
    def budget_ratio(self) -> float:
        return 1.0 - self.target

    def burn_rate(self, observed_failure_ratio: float) -> float:
        return observed_failure_ratio / max(self.budget_ratio, 1e-12)

    def budget_consumed(self, observed_failure_ratio: float,
                        duration_s: float) -> float:
        """Fraction of the whole window's budget consumed by this incident."""
        window_s = self.window_days * 86400
        return (observed_failure_ratio * duration_s) / (
            max(self.budget_ratio, 1e-12) * window_s)

    def severity(self, burn: float) -> str:
        # Multi-window multi-burn-rate thresholds, the standard SRE ladder.
        if burn >= 14.4:
            return "page immediately (budget gone in ~2 days)"
        if burn >= 6.0:
            return "page (budget gone in ~5 days)"
        if burn >= 3.0:
            return "ticket (budget gone in ~10 days)"
        if burn >= 1.0:
            return "watch (on pace to exhaust the budget)"
        return "within budget"


@dataclass
class Phase:
    """One labelled segment of an experiment timeline."""

    name: str
    start_s: float
    end_s: float = math.inf
    fault: str = ""


@dataclass
class ChaosResult:
    experiment: str = ""
    phases: list[Phase] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    events: list[tuple[float, str]] = field(default_factory=list)

    steady_ttft_p95: float = math.nan
    steady_success: float = math.nan

    fault_start_s: float = math.nan
    detect_s: float = math.nan       # first sample violating the SLO
    recover_s: float = math.nan      # first sample back within SLO, sustained
    fault_end_s: float = math.nan

    failed: int = 0
    total: int = 0
    slo_violating: int = 0

    @property
    def time_to_detect_s(self) -> float:
        return self.detect_s - self.fault_start_s if not math.isnan(self.detect_s) else math.nan

    @property
    def time_to_recover_s(self) -> float:
        return self.recover_s - self.fault_end_s if not math.isnan(self.recover_s) else math.nan

    @property
    def failure_ratio(self) -> float:
        return self.slo_violating / max(self.total, 1)

    def summary(self, slo: SLOTarget, incident_duration_s: float) -> dict[str, Any]:
        burn = slo.burn_rate(self.failure_ratio)
        return {
            "experiment": self.experiment,
            "requests": self.total,
            "hard_failures": self.failed,
            "slo_violating": self.slo_violating,
            "failure_ratio": round(self.failure_ratio, 4),
            "steady_ttft_p95_ms": round(self.steady_ttft_p95, 1),
            "time_to_detect_s": round(self.time_to_detect_s, 1),
            "time_to_recover_s": round(self.time_to_recover_s, 1),
            "burn_rate": round(burn, 2),
            "budget_consumed_pct": round(
                slo.budget_consumed(self.failure_ratio, incident_duration_s) * 100, 3),
            "severity": slo.severity(burn),
        }


# ---------------------------------------------------------------------------
# Injectors
# ---------------------------------------------------------------------------
class Injector:
    """Base injector. Subclasses implement apply/clear."""

    name = "noop"

    async def apply(self) -> str:
        return "noop"

    async def clear(self) -> str:
        return "noop"


@dataclass
class SimFault(Injector):
    """Drives the project 01 simulator's /sim/fault control plane.

    Lets the whole suite be developed and regression-tested without a GPU or a
    cluster, which matters because a chaos suite you cannot test is itself a
    liability.
    """

    url: str
    kind: FaultKind = "throttle"
    slowdown: float = 3.0
    error_rate: float = 0.0
    name: str = "sim"

    async def _post(self, body: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(f"{self.url.rstrip('/')}/sim/fault", json=body)
            return f"{r.status_code} {r.text[:120]}"

    async def apply(self) -> str:
        if self.kind == "throttle":
            return await self._post({"slowdown": self.slowdown})
        if self.kind == "pause":
            return await self._post({"paused": True})
        if self.kind == "errors":
            return await self._post({"error_rate": self.error_rate})
        return "unsupported"

    async def clear(self) -> str:
        return await self._post({"slowdown": 1.0, "paused": False,
                                 "error_rate": 0.0})


@dataclass
class GatewayBreakerFault(Injector):
    """Opens a gateway circuit breaker, simulating a provider outage."""

    gateway_url: str
    backend: str
    seconds: float = 120.0
    name: str = "gateway-breaker"

    async def apply(self) -> str:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(
                f"{self.gateway_url.rstrip('/')}/admin/backend/{self.backend}/disable",
                json={"seconds": self.seconds})
            return f"{r.status_code} {r.text[:120]}"

    async def clear(self) -> str:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(
                f"{self.gateway_url.rstrip('/')}/admin/backend/{self.backend}/disable",
                json={"seconds": 0})
            return f"{r.status_code} {r.text[:120]}"


@dataclass
class KubectlFault(Injector):
    """Real cluster faults. Requires kubectl and a live cluster.

    Blast radius is bounded by construction: the selector must match a label,
    and `max_pods` caps how many are affected. A chaos tool that can take out
    an entire deployment with a typo will not be run twice.
    """

    kind: FaultKind = "kill"
    namespace: str = "default"
    selector: str = "app=vllm"
    max_pods: int = 1
    name: str = "kubectl"
    _affected: list[str] = field(default_factory=list)

    def _pods(self) -> list[str]:
        out = subprocess.run(
            ["kubectl", "get", "pods", "-n", self.namespace, "-l", self.selector,
             "-o", "jsonpath={.items[*].metadata.name}"],
            capture_output=True, text=True, timeout=15)
        return [p for p in out.stdout.split() if p]

    async def apply(self) -> str:
        pods = self._pods()[: self.max_pods]
        if not pods:
            return f"no pods matched {self.selector} in {self.namespace}"
        self._affected = pods
        if self.kind == "kill":
            for p in pods:
                subprocess.run(["kubectl", "delete", "pod", p, "-n", self.namespace,
                                "--grace-period=0", "--force"],
                               capture_output=True, timeout=30)
            return f"killed {pods}"
        if self.kind == "partition":
            # Label the pod out of its Service's selector: it keeps running but
            # stops receiving traffic, which models a network partition more
            # faithfully than a kill.
            for p in pods:
                subprocess.run(["kubectl", "label", "pod", p, "-n", self.namespace,
                                "chaos-partitioned=true", "--overwrite"],
                               capture_output=True, timeout=15)
            return f"partitioned {pods}"
        return f"unsupported kind {self.kind}"

    async def clear(self) -> str:
        if self.kind == "partition" and self._affected:
            for p in self._affected:
                subprocess.run(["kubectl", "label", "pod", p, "-n", self.namespace,
                                "chaos-partitioned-", "--overwrite"],
                               capture_output=True, timeout=15)
            return f"restored {self._affected}"
        return "nothing to clear (killed pods are replaced by the controller)"


@dataclass
class GPUThrottleFault(Injector):
    """Real GPU throttling via nvidia-smi power limits.

    This is the closest available analogue to the thermal and power throttling
    that happens in a hot rack, and it produces genuine gray failure: the GPU
    stays healthy, passes every probe, and runs slower.

    Requires root and a GPU that permits power-limit changes. It reads the
    current limit first so `clear()` restores the real value rather than a
    guessed default.
    """

    device: int = 0
    watts: int = 150
    name: str = "gpu-throttle"
    _original: int | None = None

    def _query(self, field_name: str) -> str:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field_name}", "--format=csv,noheader,nounits",
             "-i", str(self.device)], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()

    async def apply(self) -> str:
        cur = self._query("power.limit")
        try:
            self._original = int(float(cur))
        except ValueError:
            return f"could not read current power limit: {cur!r}"
        r = subprocess.run(["nvidia-smi", "-i", str(self.device), "-pl", str(self.watts)],
                           capture_output=True, text=True, timeout=20)
        return (f"power limit {self._original}W -> {self.watts}W"
                if r.returncode == 0 else f"failed: {r.stderr[:150]}")

    async def clear(self) -> str:
        if self._original is None:
            return "no original power limit recorded"
        r = subprocess.run(
            ["nvidia-smi", "-i", str(self.device), "-pl", str(self._original)],
            capture_output=True, text=True, timeout=20)
        return (f"restored power limit to {self._original}W"
                if r.returncode == 0 else f"failed: {r.stderr[:150]}")


@dataclass
class CompositeFault(Injector):
    """Apply several faults together. Correlated failures are the realistic case."""

    faults: list[Injector] = field(default_factory=list)
    name: str = "composite"

    async def apply(self) -> str:
        return " | ".join(await asyncio.gather(*[f.apply() for f in self.faults]))

    async def clear(self) -> str:
        return " | ".join(await asyncio.gather(*[f.clear() for f in self.faults]))
