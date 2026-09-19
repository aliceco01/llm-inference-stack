# 12 - Cost-per-Token Dashboard

> Inference is a unit economics game. Whoever measures it wins.

Per-tenant, per-model token accounting with $/M tokens and utilization
tracking.

```bash
./dashboard.py report --run ../../results/demo-*.records.parquet \
    --gpu h100-sxm --model llama-3.1-8b --out results/
./dashboard.py mfu --model llama-3.1-8b --gpu h100-sxm \
    --prompt-tokens 5000000 --output-tokens 800000 --window-s 3600
./dashboard.py breakeven --gpu h100-sxm --achieved-tok-s 2500 --api-price 0.60
./dashboard.py serve --endpoints http://localhost:8000 --port 9100
```

## MFU is the wrong metric for decode

This is the technical claim this project is built around.

MFU (Model FLOPs Utilization) is the standard efficiency metric, and applying
it to decode is a category error. Decode reads the **entire weight matrix from
HBM on every step** to produce one token per sequence. Its arithmetic intensity
is therefore tiny, and its MFU is capped by physics at a few percent regardless
of how well the deployment is tuned.

Teams that report decode MFU conclude their GPUs are 97% wasted and go looking
for a bug that does not exist.

The right pairing:

| Phase | Bound by | Metric | Healthy range |
|---|---|---|---|
| prefill | compute | **MFU** | 40-60% |
| decode | memory bandwidth | **MBU** | 60-85% |

A well-tuned decode deployment sits at ~70% MBU and ~3% MFU simultaneously.
Both are true and neither is a problem. MBU is the number that actually
responds to the levers you have (batch size, quantization, KV dtype); decode
MFU responds to nothing.

`./dashboard.py mfu` reports all three, including blended MFU, which it labels
as hiding which phase is inefficient.

One accounting detail that matters: **cached prompt tokens are excluded from
the prefill FLOPs**, because they were never prefilled. Counting them would
credit the deployment for work it skipped, inflating MFU exactly when prefix
caching is working well.

## Attributing shared capacity fairly

Cost per token is trivial for a dedicated GPU. The real question is splitting
one GPU's cost across tenants, and equal-weighting tokens is simple and wrong.

The three token classes have genuinely different marginal costs:

| Class | Default weight | Why |
|---|---|---|
| cached prompt token | 0.1 | never prefilled; costs a cache lookup |
| uncached prompt token | 1.0 | one forward pass over its position |
| output token | 4.0 | a full decode step, amortised over the batch |

With equal weights, the tenant with long shared system prompts subsidises the
one generating long outputs. The weights are configurable because the right
ratio depends on your batch sizes, and the defaults are documented rather than
buried.

Cost also includes an **overhead multiplier** (default 1.35) covering CPU, RAM,
network, storage and control plane. A $/M-token figure built from the GPU line
item alone understates real cost by roughly a third.

## The breakeven question, asked honestly

```bash
./dashboard.py breakeven --gpu h100-sxm --achieved-tok-s 2500 --api-price 0.60
```

The comparison people get wrong: a self-hosted $/M-token figure computed at
100% utilization is not comparable to an API price. APIs charge per token; you
pay for the GPU whether or not it is busy.

So the tool reports the **minimum sustained utilization at which self-hosting
breaks even**, and reframes the decision: the question is not "is our per-token
cost lower" but "can we keep the fleet above that utilization", which is an
autoscaling and traffic-shaping problem (project 11), not a hardware one.

It also names what the comparison still excludes: engineering time, on-call,
model update cycles, and the reserve capacity you must hold for peak.

## The live dashboard

`./dashboard.py serve` scrapes engine `/metrics` endpoints and serves an HTML
dashboard: spend so far, $/M output tokens, token rates, prefix cache hit rate,
and prefill MFU / decode MBU side by side with decode MFU labelled as
structurally low.

It works against vLLM, SGLang and the project 01 simulator, since it reads the
standard metric names. Per-tenant splits need tenant labels on requests, so the
live view aggregates the fleet and `report` does the per-tenant breakdown from
a benchmark run's records.

## Why this connects to everything else

The cost number is downstream of every other project in this repo:

- **Project 04** (prefix caching) raises the cache hit rate, which removes
  prefill FLOPs and directly lowers $/M tokens.
- **Project 05** (quantization) changes both throughput and the memory ceiling.
- **Project 08** (chunked prefill) trades a little throughput for latency, and
  this is where you see what that cost.
- **Project 11** (autoscaling) sets utilization, which is the dominant term in
  the breakeven calculation.

A change that improves throughput but drops the cache hit rate can easily be a
net loss, and $/M tokens is the only number that sees both at once.
