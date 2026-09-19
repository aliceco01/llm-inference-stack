# Methodology

Every rule below exists because the alternative produces a number that is
wrong in a specific, predictable way. The failure mode is named in each case,
because a methodology section that only states what was done is unfalsifiable.

## 1. What is measured

**Time to first token (TTFT)** is measured from the moment request bytes are on
the wire to the arrival of the first streamed chunk **containing content**.

OpenAI-compatible servers emit a role-only delta first:

```
data: {"choices":[{"delta":{"role":"assistant"}}]}      <- arrives immediately
data: {"choices":[{"delta":{"content":"The"}}]}         <- arrives after prefill
```

Measuring to the first SSE frame measures network round-trip, not the engine.
Against the simulator in this repo with a 1024-token prompt, the two
definitions differ by 27.8 ms on a 43.5 ms TTFT: a 2.8x under-report. Both
values are recorded; the content-based one is published.

**Inter-token latency (ITL)** is the gap between successive content tokens.
Where a server coalesces several decoded tokens into one SSE frame, the gap is
amortised across the tokens it carried rather than reported as one large ITL.
Not doing this turns server-side frame batching into a fake latency spike.

**TPOT** (mean decode cost per request) is reported alongside ITL but never
instead of it. TPOT is one number per request; ITL is a distribution, and only
the distribution shows stalls from preemption and queue interference.

**Throughput** is total output tokens divided by the wall duration of the
measurement window. It is never computed as the sum of per-request rates, which
counts idle time repeatedly and inflates the result by roughly the concurrency
factor.

**Goodput** is the fraction of requests meeting both the TTFT and the ITL SLO.

## 2. Capacity is reported as goodput, not peak throughput

Raw throughput is unfalsifiable: you can always increase tokens/second by
letting latency rise without bound. The headline number in this teardown is the
**highest load at which 95% of requests meet the SLO**.

For context, in one sweep in this repo raw throughput at concurrency 64 was
2,251 tok/s against 822 tok/s at the SLO-bounded knee, a 2.7x difference. The
extra 1,429 tok/s is throughput you cannot sell.

## 3. Output length is controlled

Requests are sent with `ignore_eos` and `min_tokens` so each performs exactly
the requested number of decode steps. Without this, a model that emits EOS early
turns a "512 output token" benchmark into a 40-token benchmark and the
throughput figure stops meaning anything.

Where a backend does not support these, it is recorded in the run metadata and
cross-backend throughput comparisons from that run are treated as unreliable.

## 4. Both load models are run

**Closed loop** (fixed concurrency) caps in-flight requests by construction. A
slower server simply receives requests more slowly, so it can never be
overloaded and latency rises smoothly forever. It gives clean per-point latency
and is structurally incapable of finding a capacity limit.

**Open loop** (Poisson arrivals at a fixed rate) creates requests regardless of
completions. When service rate falls below arrival rate, the queue grows without
bound and latency diverges. This is how production traffic behaves.

Capacity claims come from open-loop runs. Closed-loop data alone cannot support
one, and a report that presents only closed-loop numbers as capacity is making
a claim its method cannot back.

## 5. Warmup is excluded and the exclusion is reported

The first requests against a fresh engine pay CUDA graph capture, lazy kernel
compilation and an empty prefix cache.

Warmup scales with concurrency: at least one full wave of `N` requests is
discarded at concurrency `N`. A fixed small warmup leaves the startup transient
in the sample at high concurrency, which produces non-monotonic load curves that
look like engine pathology and are measurement error. This was observed
directly while building the harness: a sweep with fixed 4-request warmup showed
a 610 ms p95 at concurrency 16 sitting between 75 ms at c=8 and 200 ms at c=32.

## 6. Sample sizes are large enough for the statistics reported

At least ten request-waves per concurrency level, and at least 100 successful
requests before a p99 is reported at all. Nearest-rank p99 computed from 48
samples is simply the maximum observation and carries no information about the
99th percentile. The harness flags both conditions.

Percentiles use nearest-rank, so a reported p99 is a value that actually
occurred rather than an interpolation no request experienced.

## 7. The harness checks itself

- **Little's law.** Achieved concurrency is computed by integrating in-flight
  count over time, independently of `throughput x latency`. Disagreement beyond
  15% indicates client-side queueing or a broken measurement window, and the run
  is flagged.
- **Client saturation.** Open-loop runs measure lag against the intended arrival
  schedule. If the generator falls behind, offered load was below target and the
  point is excluded rather than published.

## 8. Token counts come from the server

Throughput is derived from server-reported `usage`, which is exact and is what
providers bill on. Where a backend does not report usage, counts fall back to a
tokenizer or a character heuristic, and the run is tagged `token_source:
heuristic`. Any number derived from an estimator is labelled as such, because a
benchmark that estimates tokens and reports throughput to three significant
figures is reporting its estimator.

## 9. Provenance is mandatory

Every run records:

- git SHA and whether the working tree was dirty
- host and GPU identity, or explicitly that no GPU was present
- the model the server **actually reported** serving, not the one intended
- engine name, version and flags
- the workload fingerprint (a hash of the full length distribution and seed)
- the SLO
- whether the run was simulated

`./teardown.py verify` blocks publication of any run missing these. A benchmark
that cannot be tied to a code state and a hardware state is an anecdote.

## 10. Simulated results are labelled everywhere

Parts of this repo use a roofline-model engine simulator so the tooling can be
developed without a GPU. Simulated runs carry `simulated: true`, every chart is
watermarked `SIMULATED`, and the report leads with the disclosure rather than
burying it.

The simulator predicts timing and memory behaviour. It says nothing about output
quality and is never used to support a quality claim.

## Known limitations

- **Synthetic prompts.** Workloads are generated from a fixed vocabulary with a
  seeded RNG. This makes runs reproducible and shared prefixes byte-identical,
  and it does not reproduce the token-distribution effects of real traffic.
- **Single-region, single-tenant.** No cross-region latency, no noisy-neighbour
  contention from other workloads on the same host.
- **Cost figures use list prices** and a flat 1.35x infrastructure overhead.
  Real costs depend on commitments, spot availability and utilisation. The
  breakeven analysis in project 12 is the more honest framing.
- **Quality is not measured here.** Configurations that change numerics
  (quantization) must be paired with project 05's quality harness before any
  latency win from this teardown is acted on.
