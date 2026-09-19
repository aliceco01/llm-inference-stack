# LLM Inference Stack

Fifteen systems for serving, optimizing and scaling LLM inference, built as
**one stack rather than fifteen scripts**.

They share a single measurement core, a single memory model, and a single
engine simulator, so a latency number produced by the benchmark harness against
real vLLM and one produced by the simulator are the same type, analysed by the
same code, and comparable by construction.

```
packages/llmkit/              the shared core everything imports
  types.py                    RequestRecord: the canonical measurement
  metrics.py                  percentiles, goodput, Little's law self-check
  client.py                   instrumented OpenAI-compatible streaming client
  load.py                     closed-loop and open-loop drivers
  workload.py                 reproducible request-shape generators
  modelspec.py / gpus.py      architecture and hardware facts
  kvcache.py                  KV cache memory budgeting
  simulator/
    paged.py                  PagedAttention block allocator (real mechanism)
    cost.py                   roofline cost model
    engine.py                 continuous-batching scheduler on a virtual clock
  quant.py  specdec.py  cluster.py  autoscale.py  costs.py  gateway.py  chaos.py
```

## The projects

| # | Project | What it proves |
|---|---|---|
| [01](projects/01-inference-server) | Self-hosted inference server | continuous batching, and an endpoint everything else measures |
| [02](projects/02-benchmark-suite) | TTFT/ITL benchmark suite | load curves that cannot lie to you |
| [03](projects/03-kv-calculator) | KV cache calculator + monitor | predicting OOM instead of discovering it |
| [04](projects/04-prefix-cache-proxy) | Prefix caching proxy | routing that makes a cache actually hit |
| [05](projects/05-quantization-lab) | Quantization lab | measured quality/latency/VRAM tradeoffs |
| [06](projects/06-speculative-decoding) | Speculative decoding | acceptance-rate arithmetic and its limits |
| [07](projects/07-triton-kernels) | Triton kernels | fused softmax and RMSNorm against PyTorch |
| [08](projects/08-chunked-prefill) | Chunked prefill | decode starvation, measured |
| [09](projects/09-paged-attention) | PagedAttention deep dive | fragmentation, eviction, preemption |
| [10](projects/10-disaggregated) | Disaggregated prefill/decode | fleet shaping and the KV transfer cost |
| [11](projects/11-autoscaler) | Queue-based autoscaler | why GPU utilization is the wrong signal |
| [12](projects/12-cost-dashboard) | Cost-per-token dashboard | MFU for prefill, MBU for decode, $/M tokens |
| [13](projects/13-ai-gateway) | AI gateway | fallbacks, retry budgets, degradation chains |
| [14](projects/14-chaos-suite) | Chaos suite | gray failure, SLO burn, recovery time |
| [15](projects/15-public-teardown) | Public teardown | published curves with a provenance gate |

## Read this before the numbers

**This repo was built on an M1 MacBook Air with 8 GB of RAM and no NVIDIA GPU.**

vLLM/SGLang continuous batching, FP8/INT8/AWQ quantization, Triton kernels and
multi-GPU disaggregation all require CUDA hardware. Roughly nine of these
fifteen projects cannot produce real numbers on that machine.

Rather than pretend otherwise, the stack is built in two halves:

- **Everything is GPU-ready.** Point `--base-url` at a real vLLM or SGLang
  endpoint and the harnesses, proxy, gateway, autoscaler and chaos suite run
  unmodified. The Triton kernels are real CUDA kernels.
- **Everything is runnable today.** A faithful engine simulator (paged KV
  allocator, continuous-batching scheduler, roofline cost model) serves a real
  OpenAI-compatible HTTP API, so every component is executable and testable
  without a GPU.

Simulated runs are tagged `simulated: true`, every chart is watermarked
`SIMULATED`, and project 15's publication gate blocks anything whose provenance
is incomplete. **A benchmark teardown with fabricated numbers presented as real
would destroy the entire point of the exercise**, so the labelling is not
decorative.

The simulator predicts timing and memory behaviour. It says nothing about
output quality and is never used to support a quality claim.

**No number in this repo was measured on real GPU hardware.** See
[STATUS.md](STATUS.md) for exactly which components have been executed, which
have not, and where each quoted number came from. Publishing a repo about
measurement discipline without that page would be self-refuting.

## The simulator is a deliverable, not a fallback

Project 09 asks for a PagedAttention deep dive because "understanding the
scheduler beats memorizing flags". Building a faithful implementation of that
scheduler, with content-addressed prefix sharing, reference counting,
copy-on-write, LRU eviction and preemption by recompute or swap, demonstrates
that understanding more directly than running flags against someone else's
engine.

It is also what makes projects 3, 4, 8, 9, 11 and 14 executable on a laptop.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e packages/llmkit
pip install fastapi "uvicorn[standard]" httpx numpy matplotlib pyyaml pytest

# a real OpenAI-compatible server, simulating llama-3.1-8b on an H100
python projects/01-inference-server/mock_server.py --model llama-3.1-8b --gpu h100-sxm

# measure it
python projects/02-benchmark-suite/bench.py sweep \
    --base-url http://127.0.0.1:8000 --model llama-3.1-8b \
    --concurrency 1,2,4,8,16,32,64 --simulated --engine simulator

# these need no server at all
python projects/03-kv-calculator/kvcalc.py plan --model llama-3.1-8b --gpu h100-sxm
python projects/07-triton-kernels/reference.py verify
pytest packages/llmkit/tests -v
```

Swap in real hardware by changing one flag:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-chunked-prefill --enable-prefix-caching
python projects/02-benchmark-suite/bench.py sweep --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct --engine vllm --engine-version 0.11.0
```

## Findings that shaped the design

These came out of building and running the stack, not from reading about it.

**A naive TTFT measurement under-reports by 2.8x.** OpenAI-compatible servers
send a role-only delta before any content. Measuring to the first SSE frame
gave 15.7 ms where the first real token arrived at 43.5 ms; the 27.8 ms gap
matched the independently predicted prefill time of 27.3 ms. The client records
both definitions so the difference is visible rather than hidden.
([02](projects/02-benchmark-suite))

**An 8B model on an 80 GB H100 fits three full-context requests.** Weights take
15 GB; the remaining 49 GB of KV cache holds 401,312 tokens, which is 3.06
sequences at 128k context. The model fitting was never the question.
([03](projects/03-kv-calculator))

**Tensor parallelism stops buying KV capacity.** Llama-3.1-70B has 8 KV heads,
so per-GPU KV cost per token falls 320 -> 160 -> 80 -> 40 KiB from TP=1 to TP=8
and then stops. TP=16 doubles the GPU bill and shards no further.
([03](projects/03-kv-calculator))

**Prefill-priority scheduling starves decode completely.** A scheduler trace of
256 concurrent requests showed 6.7 seconds of consecutive prefill steps before
a single decode step ran. Every admitted sequence sat with allocated KV,
producing nothing. ([08](projects/08-chunked-prefill))

**Paging trades 50% external fragmentation for 4% internal.** The contiguous
baseline reserving `max_model_len` wasted 97.7% of reserved memory internally
and rejected a full-size request while half of memory was free. Under paging,
external fragmentation is structurally zero.
([09](projects/09-paged-attention))

**Benchmark methodology errors look exactly like engine pathology.** A sweep
with fixed 4-request warmup produced a 610 ms p95 at concurrency 16 sitting
between 75 ms at c=8 and 200 ms at c=32. The cause was startup transient, not
the engine. Warmup now scales to one full concurrency wave.
([02](projects/02-benchmark-suite))

**A control workload caught cross-contamination between experiments.** The
prefix-routing comparison ran four strategies against shared backends; the
`unique` control, which must show a 0% cache hit rate, showed 33-57% for every
strategy that ran after the first. They were inheriting a warm cache. Caches
are now reset between strategies. ([04](projects/04-prefix-cache-proxy))

**Prefix routing only matters under cache pressure.** With a full H100 KV cache
and a 40k-token working set, every replica holds every prefix and all routing
strategies score identically. The honest recommendation when your working set
fits is `least_loaded`. ([04](projects/04-prefix-cache-proxy))

**Speculative decoding is oversold by its own arithmetic.** Expected accepted
draft tokens is `alpha + alpha^2 + ... + alpha^k`, not `k * alpha`. At
alpha=0.7, k=8 that is 2.20 tokens, not 5.6. And the gain collapses at high
batch size, because the trade is FLOPs for bandwidth and a saturated server has
no spare FLOPs. ([06](projects/06-speculative-decoding))

**Decode MFU is a category error.** Decode reads the whole weight matrix per
step to emit one token per sequence, so its MFU is capped by physics at a few
percent. A healthy deployment sits at ~3% MFU and ~70% MBU simultaneously.
([12](projects/12-cost-dashboard))

## Engineering conventions

- **Every measurement records its provenance**: git SHA and dirty flag, host
  and GPU identity, the model the server *actually reported* serving, engine
  flags, workload fingerprint, SLO, and whether token counts were exact or
  estimated.
- **The harness checks itself.** Little's law is verified independently of the
  throughput calculation; client-side arrival-schedule lag is measured and
  flags runs where the load generator, not the server, was the bottleneck.
- **Capacity is goodput.** Raw throughput is unfalsifiable because you can
  always buy tokens/second with latency. The headline number is the highest
  load meeting the SLO for 95% of requests.
- **Retries are budgeted, not counted.** Per-request retry limits amplify load
  exactly when a backend is degrading.
- **Failure modes are named.** Where a design choice exists to prevent a
  specific wrong number, the comment says which one.

## Testing

```bash
pytest packages/llmkit/tests -v          # core: 60+ tests, no GPU needed
pytest projects/07-triton-kernels -v     # kernels: CUDA tests skip without a GPU
```

The test suite covers the things that were actually wrong during development:
timestamp sentinels, first-content-token semantics, multi-token frame
amortisation, Little's law consistency, GQA/MLA/sliding-window KV math, prefix
sharing and copy-on-write, and the truncated-geometric acceptance model.
