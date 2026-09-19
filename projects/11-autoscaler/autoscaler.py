#!/usr/bin/env python3
"""Queue-depth GPU autoscaler: simulate policies, then run one.

    # why GPU utilisation fails as a signal
    ./autoscaler.py signals --traffic spike --out results/

    # compare cold-start mitigations
    ./autoscaler.py policies --traffic spike --out results/

    # tune the cooldown/stabilization pair against bursty traffic
    ./autoscaler.py flapping --traffic bursts --out results/

    # emit KEDA manifests for a real cluster
    ./autoscaler.py keda --deployment vllm-llama8b --target 8

    # run the control loop directly (polls Prometheus, scales a Deployment)
    ./autoscaler.py control --prometheus http://localhost:9090 \
        --deployment vllm-llama8b --namespace default --target 8 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from llmkit import report
from llmkit.autoscale import TRAFFIC, ScalerConfig, simulate
from llmkit.prom import scrape_sync


def _plot(res, out: Path, name: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, dpi=130)
    tm = [t / 60 for t in res.t]
    axes[0].plot(tm, res.offered_rps, label="offered", color=report.PALETTE[0])
    axes[0].plot(tm, res.capacity_rps, label="capacity", color=report.PALETTE[1])
    axes[0].set_ylabel("req/s"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.25)
    axes[0].set_title(title)
    axes[1].plot(tm, res.queue, color=report.PALETTE[3])
    axes[1].set_ylabel("queue depth"); axes[1].grid(alpha=.25)
    axes[2].plot(tm, res.ready, label="ready", color=report.PALETTE[2])
    axes[2].plot(tm, res.starting, label="starting (paid, unusable)",
                 color=report.PALETTE[4], ls="--")
    axes[2].set_ylabel("replicas"); axes[2].set_xlabel("minutes")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=.25)
    fig.text(0.5, 0.5, "SIMULATED", fontsize=40, color="#D55E00", alpha=.10,
             ha="center", va="center", rotation=28, weight="bold")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def cmd_signals(args) -> int:
    """The headline: queue depth works, GPU utilisation does not."""
    out = Path(args.out)
    traffic = TRAFFIC[args.traffic]()
    rows = []
    for signal, target in (("queue_depth", 8.0), ("gpu_util", 70.0),
                           ("running_requests", 48.0), ("ttft_p95", 2000.0)):
        cfg = ScalerConfig(signal=signal, target=target,
                           min_replicas=args.min_replicas,
                           max_replicas=args.max_replicas)
        res = simulate(cfg, traffic, duration_s=args.duration,
                       capacity_per_replica_rps=args.capacity)
        s = res.summary(); s["signal"] = signal; s["target"] = target
        rows.append(s)
        _plot(res, out, f"autoscale-signal-{signal}",
              f"Scaling on {signal} (target {target})")
        print(f"{signal:<22} drops {s['drop_rate_pct']:>6.2f}%   "
              f"SLO violation {s['slo_violation_pct']:>5.1f}% of the run   "
              f"{s['gpu_hours']:>6.3f} GPU-h   {s['scale_events']:>3} events")
    print("\n" + report.md_table(rows, columns=[
        "signal", "target", "drop_rate_pct", "slo_violation_pct",
        "gpu_hours", "wasted_gpu_hours", "scale_events", "peak_replicas"]))
    print("""
GPU utilisation saturates: under continuous batching the GPU is essentially
always busy, so the metric pins near 100% whether the server is serving 10
requests or 1000. It crosses its threshold early, stops conveying information
exactly when demand keeps rising, and cannot tell you how much capacity to add.

TTFT-based scaling works but lags: latency only rises AFTER the queue is deep,
so the policy starts its multi-minute cold start after users are already
affected. It is a reasonable secondary signal and a poor primary one.

Queue depth is a direct count of arrived-but-unservable work, and it moves the
instant capacity becomes insufficient.""")
    (out / "autoscale-signals.json").write_text(json.dumps(rows, indent=2))
    return 0


def cmd_policies(args) -> int:
    """Cold-start mitigations, measured against each other."""
    out = Path(args.out)
    traffic = TRAFFIC[args.traffic]()
    variants = [
        ("reactive (baseline)", dict()),
        ("fast cold start (pre-pulled image, NVMe weights)",
         dict(image_pull_s=0.0, weight_load_s=25.0)),
        ("headroom 1.5x", dict(headroom_factor=1.5)),
        ("predictive 120s", dict(predictive_horizon_s=120.0)),
        ("headroom 1.3x + predictive 90s",
         dict(headroom_factor=1.3, predictive_horizon_s=90.0)),
        ("warm floor (min_replicas=4)", dict(min_replicas=4)),
        ("aggressive step (8/decision)", dict(max_scale_up_step=8)),
    ]
    rows = []
    for name, kw in variants:
        cfg = ScalerConfig(signal="queue_depth", target=8.0,
                           min_replicas=kw.pop("min_replicas", args.min_replicas),
                           max_replicas=args.max_replicas, **kw)
        res = simulate(cfg, traffic, duration_s=args.duration,
                       capacity_per_replica_rps=args.capacity)
        s = res.summary(); s["policy"] = name
        s["cold_start_s"] = cfg.cold_start_s
        rows.append(s)
        _plot(res, out, f"autoscale-policy-{name.split()[0].lower()}", name)
        print(f"{name:<46} drops {s['drop_rate_pct']:>6.2f}%   "
              f"SLO viol {s['slo_violation_pct']:>5.1f}%   "
              f"{s['gpu_hours']:>6.3f} GPU-h")
    print("\n" + report.md_table(rows, columns=[
        "policy", "cold_start_s", "drop_rate_pct", "slo_violation_pct",
        "gpu_hours", "wasted_gpu_hours", "peak_replicas"]))
    print("""
The tradeoff is explicit in the last two columns. Headroom and a warm floor buy
SLO at the cost of GPU-hours; shrinking cold start buys SLO for free and is
therefore the first thing to attack. In practice that means pre-pulling the
image onto the node, keeping weights on local NVMe rather than object storage,
and not paying for a fresh CUDA-graph capture on every start.

`wasted_gpu_hours` counts replicas that were paid for while still loading and
could not serve anything. It is the honest cost of reactive scaling and the
number that predictive policies are trying to avoid.""")
    (out / "autoscale-policies.json").write_text(json.dumps(rows, indent=2))
    return 0


def cmd_flapping(args) -> int:
    """Stabilization window vs responsiveness under repeated bursts."""
    out = Path(args.out)
    traffic = TRAFFIC[args.traffic]()
    rows = []
    for stab in (0, 60, 180, 300, 600):
        cfg = ScalerConfig(signal="queue_depth", target=8.0,
                           min_replicas=args.min_replicas,
                           max_replicas=args.max_replicas,
                           scale_down_stabilization_s=stab,
                           scale_down_cooldown_s=max(stab, 60))
        res = simulate(cfg, traffic, duration_s=args.duration,
                       capacity_per_replica_rps=args.capacity)
        s = res.summary(); s["scale_down_stabilization_s"] = stab
        rows.append(s)
        print(f"stabilization {stab:>4}s   events {s['scale_events']:>3}   "
              f"drops {s['drop_rate_pct']:>6.2f}%   "
              f"SLO viol {s['slo_violation_pct']:>5.1f}%   "
              f"{s['gpu_hours']:>6.3f} GPU-h")
    print("\n" + report.md_table(rows, columns=[
        "scale_down_stabilization_s", "scale_events", "drop_rate_pct",
        "slo_violation_pct", "gpu_hours"]))
    print("""
With no stabilization the policy removes a replica in every trough and pays a
full cold start in every peak, which costs both SLO and money: the worst of
both. Long stabilization keeps replicas through the troughs, spends more
GPU-hours idle, and absorbs the next burst instantly.

For inference the asymmetry is severe because scale-up is minutes and
scale-down is seconds, so scale-down should be far more conservative than
scale-up. A 5 minute stabilization window with a 30 second scale-up cooldown is
a sane starting point, which is what the KEDA manifests emit.""")
    (out / "autoscale-flapping.json").write_text(json.dumps(rows, indent=2))
    return 0


KEDA_TEMPLATE = """\
# KEDA ScaledObject for {deployment}.
#
# Scales on vllm:num_requests_waiting, which is a direct count of arrived work
# that cannot be served. Do NOT scale inference on GPU utilisation: under
# continuous batching it pins near 100% regardless of load and carries no
# information about unmet demand.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: {deployment}-scaler
  namespace: {namespace}
spec:
  scaleTargetRef:
    name: {deployment}
  minReplicaCount: {min_replicas}
  maxReplicaCount: {max_replicas}
  pollingInterval: 15
  # Long cooldown because scale-up costs minutes and scale-down costs seconds.
  # Removing a replica you need again is far more expensive than keeping it.
  cooldownPeriod: {cooldown}
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 0
          policies:
            - type: Pods
              value: 4
              periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: {stabilization}
          policies:
            - type: Pods
              value: 1
              periodSeconds: 120
  triggers:
    # Primary: queue depth per replica.
    - type: prometheus
      metadata:
        serverAddress: {prometheus}
        metricName: vllm_requests_waiting
        query: sum(vllm:num_requests_waiting{{service="{deployment}"}})
        threshold: "{target}"
    # Secondary: KV cache pressure. A replica whose cache is full will start
    # preempting even if its queue looks healthy, so this catches the memory
    # bound that queue depth alone misses.
    - type: prometheus
      metadata:
        serverAddress: {prometheus}
        metricName: vllm_kv_cache_pressure
        query: max(vllm:gpu_cache_usage_perc{{service="{deployment}"}})
        threshold: "0.9"
---
# Cold start is the dominant cost of scaling up. Everything below exists to
# shrink it, which is cheaper than provisioning headroom to hide it.
apiVersion: v1
kind: ConfigMap
metadata:
  name: {deployment}-coldstart-notes
  namespace: {namespace}
data:
  notes: |
    Measured cold start splits roughly into:
      image pull     0-60s   -> eliminate by pre-pulling onto the node pool
      weight load   30-300s  -> dominated by storage bandwidth; keep weights on
                                local NVMe, not object storage over the network
      warmup        15-45s   -> KV profiling + CUDA graph capture

    Mitigations, in order of value per unit effort:
      1. Pre-pull images onto GPU nodes (DaemonSet or node image).
      2. Cache weights on local NVMe via a PVC or an init container.
      3. Keep a warm floor: minReplicaCount above zero. Scale-to-zero is a
         false economy for anything user-facing.
      4. Over-provision modestly (headroom) rather than scaling reactively into
         a spike you cannot catch.
"""


def cmd_keda(args) -> int:
    print(KEDA_TEMPLATE.format(
        deployment=args.deployment, namespace=args.namespace,
        min_replicas=args.min_replicas, max_replicas=args.max_replicas,
        target=args.target, prometheus=args.prometheus,
        cooldown=args.cooldown, stabilization=args.stabilization))
    return 0


def cmd_control(args) -> int:
    """A real control loop. Polls Prometheus-style metrics, scales a Deployment.

    KEDA is the right tool in a real cluster; this exists so the policy can be
    run and debugged directly, and so the scaling decision is inspectable
    rather than hidden inside an operator.
    """
    cfg = ScalerConfig(signal="queue_depth", target=args.target,
                       min_replicas=args.min_replicas,
                       max_replicas=args.max_replicas,
                       scale_up_cooldown_s=args.scale_up_cooldown,
                       scale_down_cooldown_s=args.scale_down_cooldown,
                       scale_down_stabilization_s=args.stabilization)
    last_up = last_down = 0.0
    low_since: float | None = None
    print(f"controlling {args.namespace}/{args.deployment}  target={args.target} "
          f"waiting-requests per replica  dry_run={args.dry_run}\n")
    try:
        while True:
            m = scrape_sync(args.endpoint) if args.endpoint else {}
            waiting = m.get("vllm:num_requests_waiting")
            running = m.get("vllm:num_requests_running", 0.0)
            kv = m.get("vllm:gpu_cache_usage_perc", 0.0)
            if waiting is None:
                print(f"[{time.strftime('%H:%M:%S')}] no metrics from "
                      f"{args.endpoint}; is the endpoint serving /metrics?")
                time.sleep(args.interval); continue

            current = _current_replicas(args) if not args.dry_run else args.min_replicas
            per_replica = waiting / max(current, 1)
            # Same arithmetic KEDA/HPA uses: ceil(current * metric / target).
            want = math.ceil(current * max(per_replica, 0.0) / max(cfg.target, 1e-9))
            want = max(cfg.min_replicas, min(cfg.max_replicas, want))
            if kv >= 0.9 and want <= current:
                want = min(current + 1, cfg.max_replicas)
                reason = f"KV cache at {kv*100:.0f}%"
            else:
                reason = f"{per_replica:.1f} waiting/replica vs target {cfg.target}"

            now = time.time()
            action = "hold"
            if want > current and (now - last_up) >= cfg.scale_up_cooldown_s:
                action = "up"; last_up = now; low_since = None
            elif want < current:
                if low_since is None:
                    low_since = now
                elif ((now - low_since) >= cfg.scale_down_stabilization_s
                      and (now - last_down) >= cfg.scale_down_cooldown_s):
                    action = "down"; last_down = now; low_since = None
            else:
                low_since = None

            print(f"[{time.strftime('%H:%M:%S')}] waiting={waiting:>5.0f} "
                  f"running={running:>5.0f} kv={kv*100:>5.1f}% "
                  f"replicas={current} -> {want}  {action:<5} ({reason})")
            if action in ("up", "down") and not args.dry_run:
                _scale(args, want)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def _current_replicas(args) -> int:
    out = subprocess.run(
        ["kubectl", "get", "deployment", args.deployment, "-n", args.namespace,
         "-o", "jsonpath={.spec.replicas}"],
        capture_output=True, text=True, timeout=10)
    try:
        return int(out.stdout.strip() or 1)
    except ValueError:
        return 1


def _scale(args, n: int) -> None:
    subprocess.run(
        ["kubectl", "scale", "deployment", args.deployment,
         "-n", args.namespace, f"--replicas={n}"],
        capture_output=True, text=True, timeout=15)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sim(p):
        p.add_argument("--traffic", default="spike", choices=list(TRAFFIC))
        p.add_argument("--duration", type=float, default=3600.0)
        p.add_argument("--capacity", type=float, default=10.0,
                       help="requests/sec one replica can serve")
        p.add_argument("--min-replicas", type=int, default=1)
        p.add_argument("--max-replicas", type=int, default=20)
        p.add_argument("--out", default="results")

    p = sub.add_parser("signals", help="queue depth vs GPU util vs TTFT"); sim(p)
    p = sub.add_parser("policies", help="cold-start mitigations"); sim(p)
    p = sub.add_parser("flapping", help="stabilization window tuning"); sim(p)

    p = sub.add_parser("keda", help="emit KEDA manifests")
    p.add_argument("--deployment", default="vllm-llama8b")
    p.add_argument("--namespace", default="default")
    p.add_argument("--target", type=float, default=8.0)
    p.add_argument("--min-replicas", type=int, default=2)
    p.add_argument("--max-replicas", type=int, default=20)
    p.add_argument("--prometheus", default="http://prometheus:9090")
    p.add_argument("--cooldown", type=int, default=300)
    p.add_argument("--stabilization", type=int, default=300)

    p = sub.add_parser("control", help="run the control loop")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000",
                   help="an engine /metrics endpoint (or a Prometheus proxy)")
    p.add_argument("--prometheus", default=None)
    p.add_argument("--deployment", default="vllm-llama8b")
    p.add_argument("--namespace", default="default")
    p.add_argument("--target", type=float, default=8.0)
    p.add_argument("--min-replicas", type=int, default=2)
    p.add_argument("--max-replicas", type=int, default=20)
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--scale-up-cooldown", type=float, default=30.0)
    p.add_argument("--scale-down-cooldown", type=float, default=300.0)
    p.add_argument("--stabilization", type=float, default=300.0)
    p.add_argument("--dry-run", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return {"signals": cmd_signals, "policies": cmd_policies,
            "flapping": cmd_flapping, "keda": cmd_keda,
            "control": cmd_control}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
