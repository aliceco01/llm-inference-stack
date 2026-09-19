#!/usr/bin/env bash
# Launch SGLang. Same OpenAI-compatible surface as vLLM, so every harness in
# this repo points at it unchanged.
#
# SGLang's differentiator is RadixAttention: prefix sharing via a radix tree
# over the KV cache rather than a flat block hash table. It generalises better
# to branching prefixes (agent trees, beam-like exploration) where vLLM's
# hash-per-block scheme only catches linear prefixes.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"

exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --context-length "${MAX_MODEL_LEN:-8192}" \
  --mem-fraction-static "${GPU_UTIL:-0.90}" \
  `# RadixAttention prefix cache is on by default; --disable-radix-cache` \
  `# turns it off, which is how project 04 measures its contribution.` \
  --chunked-prefill-size "${MAX_BATCHED_TOKENS:-8192}" \
  --enable-metrics \
  "$@"
