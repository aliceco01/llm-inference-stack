#!/usr/bin/env bash
# Launch vLLM with continuous batching. Requires an NVIDIA GPU with CUDA.
#
# Every flag below is set deliberately. The defaults vLLM ships are good; the
# ones that need thought are called out with the reasoning, because copying a
# flag you cannot justify is how you end up with a 40% throughput regression
# you cannot explain.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"

# Sanity-check the memory budget BEFORE burning a GPU-hour discovering it
# does not fit. This is project 03 doing real work in the serving path.
python -m kvcalc plan \
  --model "${MODEL}" --gpu "${GPU:-h100-sxm}" --tp "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_UTIL}" \
  || echo "WARNING: memory pre-check failed; continuing anyway"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  \
  `# --- continuous batching: the whole point of this project ---` \
  `# max_num_seqs caps the running batch. Higher = more throughput and more` \
  `# KV pressure. 256 is vLLM's default and is right for 8B-class models;` \
  `# drop it for long-context work where 256 sequences cannot fit in cache.` \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  \
  `# max_num_batched_tokens caps tokens per step. With chunked prefill on,` \
  `# this is the knob that trades TTFT against ITL: smaller = prefill is` \
  `# split finer = decode stalls less = better ITL, slightly worse TTFT.` \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  \
  `# Chunked prefill lets prefill and decode share a step instead of prefill` \
  `# monopolising it. Measured in project 08. On by default in recent vLLM` \
  `# for most configs; set explicitly so the config is self-documenting.` \
  --enable-chunked-prefill \
  \
  `# Prefix caching reuses KV blocks across requests that share a prefix.` \
  `# Nearly free when prefixes repeat (system prompts, RAG, multi-turn) and` \
  `# close to free when they do not. Measured in project 04.` \
  --enable-prefix-caching \
  \
  `# Expose Prometheus metrics for the KV monitor and the autoscaler.` \
  --disable-log-requests \
  "$@"
