# 02 - TTFT / ITL Benchmark Suite

> Latency claims without load curves are marketing, not engineering.

A load-testing harness that measures time-to-first-token, inter-token latency
and throughput under rising concurrency, against any OpenAI-compatible endpoint
(vLLM, SGLang, TGI, hosted providers, or the simulator from project 01).

```bash
# closed-loop concurrency ladder
./bench.py sweep --base-url http://localhost:8000 --model llama-3.1-8b \
    --workload balanced --concurrency 1,2,4,8,16,32,64 --engine vllm

# open-loop: the mode that finds the capacity cliff
./bench.py rate --base-url http://localhost:8000 --model llama-3.1-8b \
    --rps 1,2,5,10,20,40 --duration 60

# one request, fully instrumented
./bench.py single --base-url http://localhost:8000 --model llama-3.1-8b
```

## Four things this harness does that most do not

### 1. TTFT is measured to the first *content* token

OpenAI-compatible servers send a role-only delta first:

```
data: {"choices":[{"delta":{"role":"assistant"}}]}       <- arrives immediately
data: {"choices":[{"delta":{"content":"The"}}]}          <- arrives after prefill
```

Measuring to the first SSE frame measures your network RTT, not the engine.
The harness records both and reports the difference. Against the project 01
server with a 1024-token prompt:

```
TTFT            43.49 ms
  first frame   15.72 ms  (delta +27.77 ms: the role-only frame)
```

27.77ms is the prefill step, and the model in `llmkit.simulator.cost`
independently predicts 27.34ms for 1024 tokens. A naive harness would have
reported this server as 2.8x faster than it is.

### 2. Open loop and closed loop are different experiments

**Closed loop** (fixed concurrency N) caps in-flight requests by construction.
A slower server simply receives requests more slowly, so it can never be
overloaded and latency rises smoothly forever. Good for clean per-point
latency; structurally incapable of finding a capacity limit.

**Open loop** (fixed arrival rate) creates requests on a Poisson schedule
regardless of completions. When service rate drops below arrival rate the
queue grows without bound and latency diverges. This is how production traffic
behaves and it is the only mode that finds the cliff.

`rate` prints offered-vs-achieved rate so saturation is explicit:

```
offered vs achieved rate (divergence = saturation):
   target rps   achieved   ratio  peak inflight
         10.0       9.98    1.00             12
         20.0      19.94    1.00             31
         40.0      28.15    0.70            412  <-- saturated
```

### 3. The harness checks itself

- **Little's law.** Achieved concurrency is measured by integrating in-flight
  count over time, independently of `throughput x latency`. If they disagree by
  more than 15% the run is flagged, because that means client-side queueing or
  a broken measurement window.
- **Client saturation.** If the load generator falls behind its own arrival
  schedule, the offered load was below target and the point is flagged for
  discard rather than published.
- **Sample size.** Warmup scales to one full concurrency wave and measured
  requests to at least ten waves. Fixed small warmup at high concurrency leaves
  the startup transient in the sample and produces non-monotonic curves that
  look like engine pathology but are measurement error. A p99 computed from 48
  samples is flagged: nearest-rank p99 on 48 points is just the maximum.

### 4. Goodput, not throughput

Throughput alone is unfalsifiable: you can always raise tok/s by letting latency
go to infinity. Goodput counts only requests meeting both the TTFT and the ITL
SLO, and the reported capacity is the highest load where 95% of requests pass:

```
SLO-bounded capacity: 822 output tok/s at concurrency 8
```

Note that raw throughput at concurrency 64 was 2,251 tok/s, 2.7x higher. That
extra throughput is unsellable: half those requests missed the SLO.

## Output length must be controlled

The harness sends `ignore_eos` and `min_tokens` so every request performs
exactly the requested number of decode steps. Without this, a model that emits
EOS early turns a "512 output token" benchmark into a 40-token benchmark and the
throughput number stops meaning anything. Pass `--no-ignore-eos` for backends
that lack these knobs, and treat cross-backend throughput comparisons as
unreliable when you do.

## Provenance

Every run records git SHA and dirty flag, host and GPU identity, the model the
server *actually reported* serving, the workload fingerprint, the SLO, and
whether token counts came from server `usage` or an estimator. Project 15
refuses to publish runs that fail this gate.
