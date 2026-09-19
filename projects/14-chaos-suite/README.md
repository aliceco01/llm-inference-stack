# 14 - Chaos Suite for Inference

> Reliability is proven under failure, not in demos.

Injects GPU throttling, replica kills, error storms and traffic spikes, then
measures SLO burn and recovery time.

```bash
./chaos_run.py list
./chaos_run.py run --scenario gray-failure \
    --target http://localhost:9000 --victim http://localhost:8001
./chaos_run.py suite --target http://localhost:9000 --victim http://localhost:8001
```

Runs against the project 13 gateway with project 01 simulator backends (no GPU,
no cluster), or against a real cluster via `--namespace` / `--selector`.

## Gray failure is the scenario that matters

The characteristic inference failure is **not a crash**. It is a replica whose
GPU is thermally throttled, or whose KV cache is thrashing, that stays up,
answers `/health` in two milliseconds, and serves tokens at a third of its
normal rate.

Every liveness probe passes. Every readiness probe passes. Kubernetes is
satisfied. The load balancer keeps sending it traffic. Users see a third of the
throughput and nothing in the orchestration layer has any idea.

This is why the gateway in project 13 routes on **TTFT SLO compliance** rather
than on health-check status, and the `gray-failure` scenario is the test that
this actually works. A system that only detects hard failures will pass every
other scenario in this suite and still degrade silently in production.

`gpu-throttle` does the same thing for real, by lowering the GPU's power limit
with `nvidia-smi -pl`. It reads the current limit first so it can restore the
real value rather than a guessed default.

## Every scenario states a hypothesis first

This follows the chaos-engineering discipline rather than "break things and
look":

1. **State a steady-state hypothesis** and measure the baseline before touching
   anything. Without a measured baseline there is no way to say whether the
   system degraded or was always like that.
2. **Inject one fault** with a bounded blast radius.
3. **Measure continuously** through fault and recovery.
4. **Report detection and recovery**, not only whether it broke.

Example:

> **gray-failure**: One replica slows to a third of its speed but stays
> "healthy".
> *Hypothesis*: the gateway detects it by TTFT SLO and shifts traffic away.
> p95 rises but stays within SLO, and no requests fail.

The run prints `HYPOTHESIS HELD` or `HYPOTHESIS REJECTED`. A rejected
hypothesis is the useful outcome; it is the only one that taught you something.

## Load must be open loop

`drive_load` uses Poisson arrivals at a fixed rate, not fixed concurrency. A
closed-loop driver reduces its own offered rate when the server slows down,
which hides exactly the degradation the experiment exists to observe. Injecting
a 3x slowdown under closed-loop load produces a graph where nothing happens.

This is the same distinction project 02 makes, and it matters more here than
anywhere else in the repo.

## Error budget accounting

Results are reported as burn rate, using the standard SRE model:

```
burn_rate = observed_failure_ratio / (1 - slo_target)
```

A burn rate of 1 is on pace to exhaust the error budget exactly at the end of
the window. 14.4 exhausts a 30-day budget in about two days, which is the
conventional page-immediately threshold. The suite maps burn rate onto that
ladder (`page immediately` / `page` / `ticket` / `watch` / `within budget`) and
reports what percentage of the whole window's budget a single incident
consumed.

This reframes the output from "did it break" to "how much of our margin did
that cost", which is the question that actually drives remediation priority.

## Detection and recovery are separate measurements

- **Time to detect**: first sample after injection whose SLO violation ratio
  exceeds the threshold.
- **Time to recover**: first run of *three consecutive* healthy samples after
  the fault is cleared. A single good sample is noise, and a suite that
  declares recovery on one is measuring luck.

A scenario with low burn but a long recovery time is still a finding: the
incident was survivable, but the system does not heal on its own.

## Scenarios

| Scenario | What it tests |
|---|---|
| `gray-failure` | SLO-based routing, the failure health checks cannot see |
| `replica-kill` | failover latency, retry safety, cold-start recovery |
| `error-storm` | circuit breaker and **retry budget** under 40% error rate |
| `provider-outage` | the degradation chain actually works |
| `traffic-spike` | autoscaler reaction vs cold start (project 11) |
| `gpu-throttle` | real thermal-style slowdown, needs a GPU and root |
| `correlated` | two faults at once, because real incidents rarely arrive alone |

`error-storm` is the one to watch `gateway_retries_total` during. If retries
scale with the error rate instead of staying inside the budget, you have
confirmed that a partial outage will become a total one, which is worth knowing
before it happens.

`correlated` exists because independent mechanisms that each work alone can
oscillate or deadlock together, and single-fault testing never finds that.

## Blast radius

`KubectlFault` requires a label selector and caps affected pods with
`max_pods` (default 1). A chaos tool that can take out an entire deployment
through a typo will be run once and then quietly disabled, which is worse than
not having one.
