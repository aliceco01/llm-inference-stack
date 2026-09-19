# 15 - Public Benchmark Teardown

> Public, reproducible benchmarks are the strongest hiring signal in infra.

Latency, throughput and cost curves for three serving configurations, with the
full methodology and a gate that refuses to publish results lacking provenance.

```bash
./teardown.py run     --config configs/teardown.yaml
./teardown.py verify  --results results/          # the publication gate
./teardown.py publish --results results/ --out site/
```

Read [METHODOLOGY.md](METHODOLOGY.md) first. It states each measurement rule
alongside the specific way the alternative produces a wrong number.

## The publication gate is the point

`verify` blocks any run that cannot be reproduced or challenged:

```
  [BLOCKED] bench-20260919-000312  engine=simulator model=llama-3.1-8b simulated=True
      - working tree dirty at run time (code state not reproducible)
      - summary warning: n=64 successful requests: p99 is effectively the
        maximum observation. At least 100 samples are needed for a meaningful p99.
```

Required: git SHA on a clean tree, the model the server **actually reported**
serving, hardware identity (including "no GPU detected"), server-reported token
counts, the workload fingerprint, the SLO, and an explicit simulated flag.

Runs can still be published with `--allow-caveats`, in which case every caveat
appears in the report body under its own heading, not in a footnote. The point
is not to make publishing hard; it is to make publishing *silently* impossible.

## One variable at a time

The three configurations differ from the baseline by one dimension each:

| Config | Change | Quality measurement required? |
|---|---|---|
| A: baseline | bf16, no prefix caching, no chunked prefill | n/a |
| B: scheduling | + prefix caching + chunked prefill | no, numerics unchanged |
| C: fp8 | B + fp8 weights and KV cache | **yes** |

A "tuned" configuration that changes six flags at once yields a number you
cannot explain and cannot act on. Config C is explicitly marked as requiring
project 05's quality harness before its latency win means anything, and the
config file says so in its own `notes` field so the constraint travels with the
data.

## What gets published

- **Latency vs load**, TTFT and ITL, p50 and p95, log axes
- **Throughput vs latency** as a parametric frontier with the SLO-bounded knee
  starred, because the tradeoff is the point and two separate charts hide it
- **Goodput**, raw throughput against throughput that actually met the SLO
- **Cost per million output tokens** computed at the SLO-bounded knee, not at
  peak throughput
- The full sweep table for every configuration
- Caveats, in the body

Charts from simulated runs are watermarked `SIMULATED` and the report leads
with the disclosure.

## The headline number

Capacity is reported as the highest load at which **95% of requests meet both
the TTFT and ITL SLO**, not as peak throughput. In one sweep in this repo, raw
throughput at concurrency 64 was 2,251 tok/s against 822 tok/s at the
SLO-bounded knee. The difference is throughput you cannot sell, and reporting
the larger number is the single most common way serving benchmarks mislead.

## Why this is the capstone

It consumes every other project:

- **02** provides the harness, the drivers and the self-checks
- **03** sizes the configurations and explains the memory limits in the results
- **04** and **08** are the changes measured in config B
- **05** supplies the quality evidence config C cannot be published without
- **12** supplies the cost basis and the breakeven framing
- **14** provides the reliability evidence that a latency curve cannot

A benchmark is only as good as what it refuses to claim, and most of the work
here is in the refusing.
