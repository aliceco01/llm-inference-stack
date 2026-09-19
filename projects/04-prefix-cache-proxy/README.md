# 04 - Prefix Caching Proxy

> Reused cache is free speed.

A gateway that routes requests sharing a system prompt to the same replica so
its KV blocks get reused instead of recomputed.

```bash
# run it
./proxy.py --backends http://localhost:8001,http://localhost:8002 \
           --router prefix_affinity --port 9000

# measure whether it helps (spins up backends + proxy locally, no GPU)
./experiment.py --replicas 3 --concurrency 12 --requests 96
```

## The mechanism

A prefix cache stores KV blocks keyed by a chained hash of their content. A
request whose first N tokens match a cached sequence skips prefill for those
tokens entirely: no compute, no new blocks. On a 2048-token shared system
prompt, that is 2048 tokens of prefill the GPU never does.

The catch is that the cache is **per replica**. Two requests with the same
system prompt sent to two different replicas each pay full prefill. Routing is
what turns a cache that exists into a cache that hits.

## Strategies implemented

Shared with project 13 in `llmkit/routing.py`.

| Router | Key | Behaviour |
|---|---|---|
| `round_robin` | none | even spread, cache-oblivious |
| `least_loaded` | none | fewest in-flight, tie-broken by queued prompt tokens |
| `prefix_affinity` | hash of system prompt + prompt head | bounded-load consistent hashing |
| `session_affinity` | session id | whole conversation history stays on one replica |

### Why bounded load, not plain affinity

Plain consistent hashing sends every request with a popular system prompt to
one replica. That maximises hit rate and saturates that replica while the rest
idle, and the resulting queueing delay is far larger than the prefill it saved.

`prefix_affinity` therefore treats the hash as a *preference*:

1. hash the prefix to a preferred replica;
2. use it if its in-flight count is within `--overload-factor` (default 1.25)
   of the fleet mean;
3. otherwise walk the ring, then fall back to least-loaded.

It also tracks which prefixes each replica has actually served recently and
prefers a demonstrated holder over the ring's guess, since the ring only
predicts where the cache *should* be.

### Why the ring uses virtual nodes

160 vnodes per replica. Adding or removing a replica then moves roughly 1/N of
keys instead of reshuffling everything, and every moved key is a cache miss on
its new replica plus a wasted cached block on its old one. Measured on the
implementation here with 8 replicas and 5000 keys:

```
remove 1 of 8: moved 14.4% of keys  (ideal 12.5%)
add 1 to 8:    moved 11.7% of keys  (ideal 11.1%)
load spread across replicas: 25%
```

## Two methodology traps this experiment hit

Both were caught by the `unique` control workload, which has no shared prefix
and must therefore show 0% hit rate under every strategy. Any strategy that
beats round-robin there is measuring an artifact.

**1. Cross-contamination between strategies.** The first version of
`experiment.py` ran all four routers against the same backend processes. The
control immediately showed 0% for round-robin (which ran first) and 33-57% for
the three that ran after it: they were inheriting a cache that round-robin had
warmed. Fixed by `POST /sim/reset` on every backend between strategies.

**2. Cache large enough to make routing irrelevant.** With a full H100 KV cache
(~400k tokens) and a working set of ~40k tokens, every replica holds every
prefix, so routing cannot change the hit rate and all four strategies score
identically. Prefix routing is only worth building when the working set exceeds
per-replica cache capacity. `experiment.py` therefore constrains backends with
`--blocks 768` (12,288 tokens) to put the system in that regime deliberately.

The second point is the practically important one: **before building
prefix-aware routing, check whether your working set actually exceeds one
replica's cache.** If it does not, this project buys nothing and the honest
answer is to use `least_loaded`.

## Running the experiment

`experiment.py` spins up N simulated backends and one proxy, then drives three
workloads through each strategy with the cache reset in between:

- `rag` - a few large system prompts shared by many requests
- `multiturn` - conversations whose growing history is fully reused
- `unique` - the control, no shared prefix

It writes `results/prefix-routing-report.md` plus per-workload charts. All
numbers it produces are simulated and watermarked as such: they come from the
roofline model in `llmkit.simulator.cost`, not from hardware. To get real
numbers, point `--backends` at real vLLM replicas instead.

## Operational notes

- **Retries are safe only before the first byte.** `forward()` opens the
  upstream stream and checks the status before returning a `StreamingResponse`,
  so a 5xx or connection failure can be retried on another replica. Once tokens
  have been forwarded, retrying would duplicate output, so it does not.
- **Circuit breaker on consecutive failures.** Without it, one wedged replica
  makes every request pay a timeout before failing over, converting one bad
  backend into fleet-wide latency.
- **Passive health matters more than active.** A replica can answer `/health`
  while its engine is wedged, so request outcomes drive the breaker and the
  `/health` poll only handles clean removals.
- **`POST /admin/router`** switches strategy without a restart, so you can A/B
  routing against live traffic.
