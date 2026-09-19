# 11 - Queue-Based GPU Autoscaler

> GPUs idling at 30% utilization burn money. Autoscaling is FinOps.

KEDA-style scaling on pending-request queue depth, with cold-start mitigation,
plus a simulator that evaluates policies before you deploy one.

```bash
./autoscaler.py signals  --traffic spike --out results/   # why GPU util fails
./autoscaler.py policies --traffic spike --out results/   # cold-start mitigations
./autoscaler.py flapping --traffic bursts --out results/  # stabilization tuning
./autoscaler.py keda --deployment vllm-llama8b --target 8 # emit manifests
./autoscaler.py control --endpoint http://localhost:8000 --dry-run
```

## Do not scale inference on GPU utilization

This is the central point. Under continuous batching the GPU is essentially
always busy, so `nvidia-smi` utilization pins near 100% whether the server is
handling 10 requests or 1000. The metric measures "was a kernel resident during
the sampling window", not "how much work is waiting".

It therefore crosses any sensible threshold early, then stops conveying
information exactly when demand keeps climbing, and it can never tell you *how
much* capacity to add. `./autoscaler.py signals` models this failure directly
against three alternatives.

| Signal | Problem |
|---|---|
| GPU utilization | saturates at ~100% long before the server does; no headroom information |
| TTFT p95 | correct but lagging: latency rises only after the queue is deep, so the multi-minute cold start starts after users are already hurting |
| running requests | capped by `max_num_seqs`, so it also saturates |
| **queue depth** | direct count of arrived-but-unservable work; moves the instant capacity becomes insufficient |

`vllm:num_requests_waiting` is the signal. The manifests pair it with a
secondary trigger on `vllm:gpu_cache_usage_perc`, because a replica whose KV
cache is full will start preempting (project 09) even while its queue looks
healthy, and that is a capacity problem queue depth alone will not catch.

## Cold start is the real constraint

A replica takes minutes to become useful. Reactive scaling therefore starts a
cold start when the queue builds and lands the new capacity after the spike has
passed. The simulator splits cold start into its components because that
determines which part you can actually fix:

```
image pull     0-60s    -> eliminate by pre-pulling onto the GPU node pool
weight load   30-300s   -> storage bandwidth bound; local NVMe, not object storage
warmup        15-45s    -> KV profiling + CUDA graph capture
```

`./autoscaler.py policies` compares mitigations and reports both sides of the
trade: SLO violation *and* GPU-hours.

- **Shrinking cold start buys SLO for free** and is the first thing to attack.
- **Headroom and a warm floor buy SLO with money.** They work; they are just
  not free, and the simulator prices them.
- **Predictive scaling** extrapolates the queue's slope to start replicas
  early. It recovers part of the cold-start delay and its failure mode
  (overshooting on a spike that stops) shows up as `wasted_gpu_hours`.

`wasted_gpu_hours` counts replicas being paid for while still loading and unable
to serve anything. It is the honest cost of reactive scaling.

**Scale-to-zero is a false economy for anything user-facing.** The first
request after a scale-to-zero pays the full cold start, which for a 70B model is
minutes. Keep a warm floor.

## Scale down slowly, scale up fast

The asymmetry is severe: scale-up costs minutes, scale-down costs seconds. So
removing a replica you turn out to need is far more expensive than keeping one
you did not.

`./autoscaler.py flapping` demonstrates this against repeated bursts. With no
stabilization window the policy removes a replica in every trough and pays a
full cold start in every peak, losing on both SLO and cost simultaneously. The
emitted KEDA config uses a 0 second scale-up stabilization with a 300 second
scale-down stabilization for exactly this reason.

## Running it

`./autoscaler.py keda` emits a `ScaledObject` with the queue-depth trigger, the
KV-pressure secondary trigger, and the asymmetric HPA behaviour block. That is
what belongs in a real cluster.

`./autoscaler.py control` runs the same policy directly as a loop: it scrapes
an engine's `/metrics`, applies `ceil(current * waiting_per_replica / target)`
(the same arithmetic KEDA and the HPA use), respects cooldowns and the
stabilization window, and calls `kubectl scale`. It exists so the decision is
inspectable and debuggable rather than hidden inside an operator, and it
supports `--dry-run` so you can watch what it *would* do against production
metrics before letting it act.

## Choosing the target

The target is waiting requests per replica. Too low and you scale on noise; too
high and the queue is already painful before you react. Derive it from the SLO
rather than guessing: if a replica serves R requests/second and your TTFT budget
allows T seconds of queueing, the queue depth that consumes that budget is
`R * T`, and the target should sit meaningfully below it. Project 02's load
curves give you R at your SLO, which is the point of measuring the knee.
