# 01 - Self-Hosted Inference Server

> You cannot optimize what you have never served.

Serves an open model behind an OpenAI-compatible API with continuous batching
enabled, and provides the endpoint every other project in this repo measures.

## Two ways to run it

**Real GPU (vLLM or SGLang).** Requires NVIDIA hardware.

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct TP=1 ./serve_vllm.sh
MODEL=meta-llama/Llama-3.1-8B-Instruct TP=1 ./serve_sglang.sh
```

**No GPU (simulator).** A real async HTTP server whose token timing comes from
the roofline model in `llmkit.simulator.cost`:

```bash
python mock_server.py --model llama-3.1-8b --gpu h100-sxm --max-model-len 4096
```

Both speak the same API, expose the same Prometheus metric names, and are
interchangeable to every harness in this repo. The simulator exists so that
gateway logic, autoscaling, chaos testing and the benchmark suite can be
developed and regression-tested without a GPU. It emits filler tokens and is
tagged `simulated: true` in `/v1/models`. **Never make a quality claim from it.**

## What continuous batching actually is

The naive server processes one batch to completion before starting the next, so
a request that finishes at step 10 waits idle until the slowest member of its
batch finishes at step 500. GPU utilisation collapses and tail latency is set by
the longest request in each batch.

Continuous batching (aka iteration-level scheduling) instead re-forms the batch
every decode step. A finished sequence leaves immediately and a waiting one
takes its slot in the next step. Two consequences:

1. Throughput rises several-fold at the same latency, because the batch stays
   full instead of draining.
2. The unit of scheduling becomes the *step*, not the *request*, which is why
   every latency metric in this repo is derived from token arrival times rather
   than request duration.

Verify it is on by watching `vllm:num_requests_running` while requests of very
different output lengths are in flight. Under continuous batching the number
stays near `max_num_seqs`; under static batching it sawtooths to zero.

## Flags that actually matter

| Flag | Effect | How to choose |
|---|---|---|
| `--max-num-seqs` | cap on running batch | Raise for throughput until KV cache is the limit. Check with project 03. |
| `--max-num-batched-tokens` | tokens per step | With chunked prefill, the TTFT/ITL dial. Measured in project 08. |
| `--enable-chunked-prefill` | mix prefill and decode | Protects ITL when long prompts arrive. Project 08. |
| `--enable-prefix-caching` | reuse KV across requests | Large win when prefixes repeat. Project 04. |
| `--gpu-memory-utilization` | fraction of VRAM vLLM may use | 0.90 default. Raising it to 0.95 buys KV cache and risks OOM under fragmentation. |
| `--tensor-parallel-size` | shard across GPUs | Needed when weights do not fit. Note KV cache stops shrinking once TP exceeds `num_kv_heads`: project 03 flags this. |

## Kubernetes

`k8s/deployment.yaml` is production-shaped. Three things in it are load-bearing
and commonly missed:

- **`/dev/shm` sized to 16Gi.** The 64MB default deadlocks NCCL during
  multi-GPU init, and it presents as a hang rather than an error.
- **Liveness far more tolerant than readiness.** A busy engine under a long
  prefill can miss probes. Restarting it converts a latency blip into a
  multi-minute outage while the model reloads.
- **`preStop` sleep + 120s grace.** In-flight generations need to drain, or
  every rollout drops active streams.
