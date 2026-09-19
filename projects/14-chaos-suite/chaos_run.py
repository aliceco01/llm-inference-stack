#!/usr/bin/env python3
"""Chaos experiments for inference: inject, measure SLO burn, measure recovery.

    # list the scenarios
    ./chaos_run.py list

    # against the project 01 simulator (no GPU, no cluster)
    ./chaos_run.py run --scenario gray-failure \
        --target http://localhost:9000 --victim http://localhost:8001

    # against a real cluster
    ./chaos_run.py run --scenario replica-kill \
        --target http://gateway/ --namespace prod --selector app=vllm

    # every scenario, one report
    ./chaos_run.py suite --target http://localhost:9000 --victim http://localhost:8001

Structure follows the chaos-engineering discipline rather than "break things
and look":

1. **State a steady-state hypothesis** and measure it before touching anything.
   Without a measured baseline there is no way to say whether the system
   degraded or was always like that.
2. **Inject one fault**, with a bounded blast radius.
3. **Measure continuously** through fault and recovery, not just at the end.
4. **Report detection and recovery times**, not only whether it broke. Recovery
   behaviour is the part that actually differs between systems.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmkit import (
    SLO,
    EndpointConfig,
    Request,
    StreamingClient,
    WorkloadGenerator,
    report,
)
from llmkit.chaos import (
    ChaosResult,
    CompositeFault,
    GatewayBreakerFault,
    GPUThrottleFault,
    Injector,
    KubectlFault,
    SimFault,
    SLOTarget,
)
from llmkit.metrics import percentile
from llmkit.workload import PRESETS, LengthSpec, WorkloadSpec


# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    description: str
    hypothesis: str
    make_fault: Callable[[argparse.Namespace], Injector]
    fault_duration_s: float = 60.0
    notes: str = ""


SCENARIOS: dict[str, Scenario] = {}


def scenario(s: Scenario) -> Scenario:
    SCENARIOS[s.name] = s
    return s


scenario(Scenario(
    name="gray-failure",
    description="One replica slows to a third of its speed but stays 'healthy'",
    hypothesis=("The gateway detects the slow replica by its TTFT SLO and shifts "
                "traffic away. p95 rises but stays within SLO, and no requests fail."),
    make_fault=lambda a: SimFault(a.victim, kind="throttle", slowdown=a.slowdown),
    fault_duration_s=90.0,
    notes=("The most important scenario in the suite. Liveness and readiness "
           "probes both pass, so every orchestrator-level mechanism is blind. "
           "Only request-outcome-driven routing can see this."),
))

scenario(Scenario(
    name="replica-kill",
    description="A replica is killed without warning",
    hypothesis=("In-flight requests on the victim fail or are retried; new "
                "requests route elsewhere within one health interval. Recovery "
                "completes when the replacement finishes its cold start."),
    make_fault=lambda a: (KubectlFault(kind="kill", namespace=a.namespace,
                                       selector=a.selector, max_pods=a.max_pods)
                          if a.namespace else SimFault(a.victim, kind="pause")),
    fault_duration_s=60.0,
    notes="Measures failover latency and whether retries are safe.",
))

scenario(Scenario(
    name="error-storm",
    description="A backend starts returning 503 for 40% of requests",
    hypothesis=("The circuit breaker opens after consecutive failures and the "
                "retry budget prevents load amplification onto healthy replicas."),
    make_fault=lambda a: SimFault(a.victim, kind="errors", error_rate=0.4),
    fault_duration_s=60.0,
    notes=("Watch gateway_retries_total. If retries scale with the error rate "
           "rather than staying inside the budget, a partial outage will become "
           "a total one."),
))

scenario(Scenario(
    name="provider-outage",
    description="The primary backend is removed entirely",
    hypothesis=("The gateway degrades to the next tier. Requests keep succeeding "
                "with higher latency, and x-gateway-degraded flips to true."),
    make_fault=lambda a: GatewayBreakerFault(a.target, a.backend_name,
                                             seconds=a.fault_duration),
    fault_duration_s=60.0,
    notes="Verifies the degradation chain actually works before you need it.",
))

scenario(Scenario(
    name="traffic-spike",
    description="Offered load jumps 5x with no warning",
    hypothesis=("Queue depth rises, the autoscaler reacts, and TTFT recovers "
                "within one cold-start period. Some SLO burn is expected and "
                "acceptable; sustained burn is not."),
    make_fault=lambda a: SimFault(a.victim, kind="throttle", slowdown=1.0),
    fault_duration_s=90.0,
    notes=("The fault here is the load itself, applied by the driver. Cold start "
           "(project 11) sets the recovery floor."),
))

scenario(Scenario(
    name="gpu-throttle",
    description="Real GPU power limit reduced, producing thermal-style slowdown",
    hypothesis=("Token rate falls, ITL rises, health checks stay green. The "
                "system should route away on latency, not wait for a failure."),
    make_fault=lambda a: GPUThrottleFault(device=a.gpu_device, watts=a.watts),
    fault_duration_s=90.0,
    notes="Requires root and a real GPU. The closest analogue to a hot rack.",
))

scenario(Scenario(
    name="correlated",
    description="A slow replica and an error storm at the same time",
    hypothesis=("Degradation is graceful under correlated failure: the breaker "
                "and the SLO-aware routing do not fight each other."),
    make_fault=lambda a: CompositeFault(faults=[
        SimFault(a.victim, kind="throttle", slowdown=a.slowdown),
        SimFault(a.victim2 or a.victim, kind="errors", error_rate=0.3),
    ]),
    fault_duration_s=90.0,
    notes=("Real incidents are rarely single-fault. Independent mechanisms that "
           "each work alone can deadlock or oscillate together."),
))


# ---------------------------------------------------------------------------
async def drive_load(ep: EndpointConfig, spec: WorkloadSpec, *, rps: float,
                     duration_s: float, slo: SLO, result: ChaosResult,
                     t0: float, sample_s: float = 2.0,
                     spike_at_s: float | None = None,
                     spike_factor: float = 5.0) -> None:
    """Open-loop load with periodic sampling.

    Open loop is required here: a closed-loop driver reduces its own offered
    rate when the server slows down, which hides exactly the degradation the
    experiment is trying to observe.
    """
    gen = WorkloadGenerator(spec)
    bucket: list[Any] = []
    lock = asyncio.Lock()

    async def sampler() -> None:
        while True:
            await asyncio.sleep(sample_s)
            async with lock:
                batch, bucket[:] = list(bucket), []
            t = time.monotonic() - t0
            if not batch:
                result.samples.append({"t": t, "n": 0, "ttft_p95": math.nan,
                                       "success": math.nan, "violating": 0})
                continue
            ok = [r for r in batch if r.ok]
            ttfts = [r.ttft_ms for r in ok if not math.isnan(r.ttft_ms)]
            p95 = percentile(ttfts, 95) if ttfts else math.nan
            viol = sum(0 if slo.met_by(r) else 1 for r in batch)
            result.total += len(batch)
            result.failed += len(batch) - len(ok)
            result.slo_violating += viol
            result.samples.append({
                "t": t, "n": len(batch),
                "ttft_p50": percentile(ttfts, 50) if ttfts else math.nan,
                "ttft_p95": p95,
                "success": len(ok) / len(batch),
                "violating": viol,
                "violation_ratio": viol / len(batch),
            })

    sampler_task = asyncio.create_task(sampler())
    async with StreamingClient(ep) as client:
        end = time.monotonic() + duration_s
        rng = random.Random(17)
        tasks: set[asyncio.Task] = set()

        async def fire(r: Request) -> None:
            rec = await client.send(r)
            async with lock:
                bucket.append(rec)

        while time.monotonic() < end:
            cur_rps = rps
            if spike_at_s is not None:
                elapsed = time.monotonic() - t0
                if spike_at_s <= elapsed < spike_at_s + 60:
                    cur_rps = rps * spike_factor
            await asyncio.sleep(rng.expovariate(cur_rps))
            req = next(iter(gen.stream(1)))
            t = asyncio.create_task(fire(req))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.wait(tasks, timeout=60)
    sampler_task.cancel()
    try:
        await sampler_task
    except asyncio.CancelledError:
        pass


def analyse(result: ChaosResult, slo_violation_threshold: float = 0.05,
            sustained_samples: int = 3) -> None:
    """Derive detection and recovery times from the sample series."""
    steady = [s for s in result.samples
              if s["t"] < result.fault_start_s and s["n"] > 0]
    if steady:
        result.steady_ttft_p95 = percentile(
            [s["ttft_p95"] for s in steady if not math.isnan(s["ttft_p95"])], 95)
        result.steady_success = sum(s["success"] for s in steady) / len(steady)

    # Detection: first sample after injection whose violation ratio exceeds
    # the threshold.
    for s in result.samples:
        if s["t"] >= result.fault_start_s and s["n"] > 0:
            if s.get("violation_ratio", 0) > slo_violation_threshold:
                result.detect_s = s["t"]
                break

    # Recovery: first run of `sustained_samples` consecutive healthy samples
    # after the fault was cleared. A single good sample is noise.
    run = 0
    for s in result.samples:
        if s["t"] < result.fault_end_s or s["n"] == 0:
            continue
        if s.get("violation_ratio", 1.0) <= slo_violation_threshold:
            run += 1
            if run >= sustained_samples:
                result.recover_s = s["t"]
                break
        else:
            run = 0


def plot(result: ChaosResult, out: Path, slo: SLO) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, dpi=130)
    t = [s["t"] for s in result.samples]
    axes[0].plot(t, [s.get("ttft_p50", math.nan) for s in result.samples],
                 label="p50", color=report.PALETTE[0])
    axes[0].plot(t, [s["ttft_p95"] for s in result.samples],
                 label="p95", color=report.PALETTE[1])
    axes[0].axhline(slo.ttft_ms, color="#D55E00", ls=":", lw=1.4, label="SLO")
    axes[0].set_ylabel("TTFT (ms)"); axes[0].set_yscale("log")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.25)
    axes[0].set_title(f"chaos: {result.experiment}")

    axes[1].plot(t, [s["success"] * 100 if not math.isnan(s["success"]) else math.nan
                     for s in result.samples], color=report.PALETTE[2])
    axes[1].set_ylabel("success %"); axes[1].set_ylim(-2, 102); axes[1].grid(alpha=.25)

    axes[2].plot(t, [s.get("violation_ratio", math.nan) * 100
                     for s in result.samples], color=report.PALETTE[3])
    axes[2].set_ylabel("SLO violation %"); axes[2].set_xlabel("seconds")
    axes[2].grid(alpha=.25)

    for ax in axes:
        ax.axvspan(result.fault_start_s, result.fault_end_s,
                   color="#D55E00", alpha=.10)
        if not math.isnan(result.detect_s):
            ax.axvline(result.detect_s, color="#E69F00", ls="--", lw=1)
        if not math.isnan(result.recover_s):
            ax.axvline(result.recover_s, color="#009E73", ls="--", lw=1)
    fig.text(0.5, 0.5, "SIMULATED", fontsize=40, color="#D55E00", alpha=.08,
             ha="center", va="center", rotation=28, weight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
async def run_scenario(sc: Scenario, args) -> ChaosResult:
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)
    slo_target = SLOTarget(target=args.slo_target, window_days=args.slo_window_days)
    ep = EndpointConfig(base_url=args.target, model=args.model)
    spec = PRESETS.get(args.workload) or WorkloadSpec(
        name="chaos", input_len=LengthSpec("fixed", 512),
        output_len=LengthSpec("fixed", 64))

    res = ChaosResult(experiment=sc.name)
    fault = sc.make_fault(args)
    t0 = time.monotonic()
    res.fault_start_s = args.steady_s
    res.fault_end_s = args.steady_s + (args.fault_duration or sc.fault_duration_s)
    total = res.fault_end_s + args.recovery_s

    print(f"\n=== {sc.name} ===")
    print(f"  {sc.description}")
    print(f"  hypothesis: {sc.hypothesis}")
    if sc.notes:
        print(f"  note: {sc.notes}")
    print(f"  timeline: {args.steady_s:.0f}s steady -> "
          f"{res.fault_end_s - res.fault_start_s:.0f}s fault -> "
          f"{args.recovery_s:.0f}s recovery")

    spike_at = res.fault_start_s if sc.name == "traffic-spike" else None
    load = asyncio.create_task(drive_load(
        ep, spec, rps=args.rps, duration_s=total, slo=slo, result=res, t0=t0,
        sample_s=args.sample_s, spike_at_s=spike_at,
        spike_factor=args.spike_factor))

    async def at(ts: float, coro, label: str) -> None:
        await asyncio.sleep(max(0.0, ts - (time.monotonic() - t0)))
        msg = await coro()
        res.events.append((time.monotonic() - t0, f"{label}: {msg}"))
        print(f"  [{time.monotonic()-t0:6.1f}s] {label}: {msg}")

    await asyncio.gather(
        load,
        at(res.fault_start_s, fault.apply, "INJECT"),
        at(res.fault_end_s, fault.clear, "CLEAR"),
    )

    analyse(res, slo_violation_threshold=args.violation_threshold)
    s = res.summary(slo_target, incident_duration_s=res.fault_end_s - res.fault_start_s)
    print(f"\n  steady TTFT p95      {s['steady_ttft_p95_ms']} ms")
    print(f"  requests             {s['requests']} "
          f"({s['hard_failures']} hard failures, {s['slo_violating']} SLO violations)")
    print(f"  time to detect       {s['time_to_detect_s']} s")
    print(f"  time to recover      {s['time_to_recover_s']} s")
    print(f"  error budget burn    {s['burn_rate']}x -> {s['severity']}")
    print(f"  budget consumed      {s['budget_consumed_pct']}% of the "
          f"{args.slo_window_days:.0f}-day budget")

    verdict = "HYPOTHESIS HELD" if (
        s["failure_ratio"] <= args.violation_threshold * 2
        and not math.isnan(res.time_to_recover_s)) else "HYPOTHESIS REJECTED"
    print(f"  {verdict}")
    res.samples.append({"verdict": verdict})

    if args.out:
        out = Path(args.out)
        plot(res, out / f"chaos-{sc.name}.png", slo)
        (out / f"chaos-{sc.name}.json").write_text(json.dumps({
            "summary": s, "hypothesis": sc.hypothesis, "verdict": verdict,
            "events": res.events, "samples": res.samples}, indent=2, default=str))
    return res


def cmd_list(args) -> int:
    print("scenarios:\n")
    for s in SCENARIOS.values():
        print(f"  {s.name:<18} {s.description}")
        print(f"  {'':<18} hypothesis: {s.hypothesis}")
        if s.notes:
            print(f"  {'':<18} note: {s.notes}")
        print()
    return 0


async def cmd_run(args) -> int:
    sc = SCENARIOS.get(args.scenario)
    if not sc:
        raise SystemExit(f"unknown scenario {args.scenario}. "
                         f"Known: {', '.join(SCENARIOS)}")
    await run_scenario(sc, args)
    return 0


async def cmd_suite(args) -> int:
    results = []
    names = (args.scenarios.split(",") if args.scenarios
             else ["gray-failure", "error-storm", "replica-kill", "traffic-spike"])
    slo_target = SLOTarget(target=args.slo_target, window_days=args.slo_window_days)
    for name in names:
        sc = SCENARIOS.get(name.strip())
        if not sc:
            print(f"skipping unknown scenario {name}")
            continue
        res = await run_scenario(sc, args)
        results.append(res.summary(
            slo_target, incident_duration_s=res.fault_end_s - res.fault_start_s))
        await asyncio.sleep(args.gap_s)

    print("\n" + report.md_table(results, columns=[
        "experiment", "requests", "hard_failures", "failure_ratio",
        "time_to_detect_s", "time_to_recover_s", "burn_rate", "severity"]))
    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        md = ["# Chaos suite results", "",
              f"- target: `{args.target}`",
              f"- SLO: TTFT <= {args.slo_ttft:.0f}ms, availability "
              f"{args.slo_target*100:.1f}% over {args.slo_window_days:.0f} days", "",
              report.md_table(results), "",
              "## Reading burn rate", "",
              "Burn rate is the multiple of budget consumption that would exactly",
              "exhaust the error budget over the SLO window. 1x is on pace to",
              "exhaust it; 14.4x exhausts a 30-day budget in about 2 days and is",
              "the conventional page-immediately threshold.", "",
              "A scenario with low burn but a long recovery time is still a",
              "finding: it means the incident was survivable but the system does",
              "not heal on its own.", ""]
        (out / "chaos-suite-report.md").write_text("\n".join(md))
        print(f"\nwrote {out}/chaos-suite-report.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list scenarios")

    def common(p):
        p.add_argument("--target", default="http://127.0.0.1:9000",
                       help="gateway or engine under test")
        p.add_argument("--victim", default="http://127.0.0.1:8001",
                       help="the backend to fault (simulator control plane)")
        p.add_argument("--victim2", default=None)
        p.add_argument("--model", default="llama-3.1-8b")
        p.add_argument("--workload", default="short")
        p.add_argument("--rps", type=float, default=8.0)
        p.add_argument("--steady-s", type=float, default=40.0)
        p.add_argument("--fault-duration", type=float, default=0.0)
        p.add_argument("--recovery-s", type=float, default=60.0)
        p.add_argument("--sample-s", type=float, default=2.0)
        p.add_argument("--slo-ttft", type=float, default=1500.0)
        p.add_argument("--slo-itl", type=float, default=60.0)
        p.add_argument("--slo-target", type=float, default=0.99)
        p.add_argument("--slo-window-days", type=float, default=30.0)
        p.add_argument("--violation-threshold", type=float, default=0.05)
        p.add_argument("--slowdown", type=float, default=3.0)
        p.add_argument("--spike-factor", type=float, default=5.0)
        p.add_argument("--backend-name", default="vllm-a")
        p.add_argument("--namespace", default=None)
        p.add_argument("--selector", default="app=vllm")
        p.add_argument("--max-pods", type=int, default=1)
        p.add_argument("--gpu-device", type=int, default=0)
        p.add_argument("--watts", type=int, default=150)
        p.add_argument("--out", default="results")

    p = sub.add_parser("run", help="one scenario"); common(p)
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    p = sub.add_parser("suite", help="several scenarios"); common(p)
    p.add_argument("--scenarios", default=None)
    p.add_argument("--gap-s", type=float, default=15.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    return asyncio.run({"run": cmd_run, "suite": cmd_suite}[args.cmd](args))


if __name__ == "__main__":
    raise SystemExit(main())
