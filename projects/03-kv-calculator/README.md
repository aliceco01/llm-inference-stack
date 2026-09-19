# 03 - KV Cache Memory Calculator and Monitor

> Most production OOMs are KV cache math errors, not model size errors.

Two tools: `kvcalc.py` predicts KV cache VRAM for a model and config before you
deploy, and `kvmon.py` watches live cache utilisation on a running engine.

## The arithmetic

```
KV bytes per token = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes
```

The field that matters is **`num_key_value_heads`, not `num_attention_heads`**.
Grouped-query attention makes them differ by up to 8x:

```bash
./kvcalc.py config --path ./Llama-3.1-70B/config.json
```
```
llama-3.1-70b: 80L h=8192 heads=64/kv=8 (gqa, GQA ratio 8x) head_dim=128
  KV bytes/token   fp16 320.00 KiB   fp8 160.00 KiB

  NOTE: sizing KV off num_attention_heads instead of num_key_value_heads
  would give 2560.0 KiB/token, a 8x overestimate.
```

Multi-head latent attention (DeepSeek-V3) breaks the formula entirely: it caches
one compressed latent per layer with no K/V factor of 2, so a 671B model has a
*smaller* per-token cache (68.6 KiB) than an 8B GQA model (128 KiB).

## Why the model fitting is not the question

```bash
./kvcalc.py plan --model llama-3.1-8b --gpu h100-sxm --max-model-len 131072
```
```
  GPU memory (total)                     74.51 GiB
  x gpu_memory_utilization 0.9           67.06 GiB
  - model weights (bf16)                -14.96 GiB
  - peak activations                     -0.81 GiB
  - CUDA graphs                          -1.50 GiB
  - framework/NCCL overhead              -0.80 GiB
  = KV cache available                   48.99 GiB

  Blocks                 25,082
  Cacheable tokens      401,312

  full-context (128k) seqs that fit: 3
```

An 8B model on an 80 GB H100 uses 15 GB for weights. The remaining 49 GB of KV
cache holds exactly **three** full-context sequences. The model fits trivially;
the context is what kills you, and only once real traffic shows up.

## Predicting the OOM before it happens

```bash
./kvcalc.py oom --model llama-3.1-8b --gpu h100-sxm --concurrency 64 \
    --median-seq-len 1024 --p95-seq-len 8192 --p99-seq-len 32768
```
```
      case   seq len  x concurrency   tokens needed   headroom   verdict
    median     1,024             64          65,536      6.12x        OK
       p95     8,192             64         524,288      0.77x       OOM
       p99    32,768             64       2,097,152      0.19x       OOM
```

This is the shape of nearly every KV incident: capacity planned against median
prompt length, 6x headroom on the dashboard, and the p95 of real traffic does
not fit. Note the failure mode is **not** an OOM traceback. vLLM preempts
instead, so you see a latency cliff and a rising preemption counter, which is
why it gets misdiagnosed as "the model got slower".

Exit code is non-zero when it predicts OOM, so it works as a CI gate.

## Tensor parallelism stops buying KV cache

```bash
./kvcalc.py tp-scan --model llama-3.1-70b --gpu h100-sxm --seq-len 32768
```
```
    TP   weights/GPU   KV/tok/GPU  KV GiB/GPU  total KV GiB    seqs@32768
     1       131.50G       320.0K          --            --    weights do not fit
     2        65.75G       160.0K          --            --    weights do not fit
     4        32.88G        80.0K       30.44        121.75            12
     8        16.44G        40.0K       46.99        375.89            37
    16         8.22G        40.0K       55.26        884.15            44  <-- KV/token stopped shrinking (KV heads replicated)
```

Weights shard cleanly with TP forever. KV heads only shard until TP reaches
`num_kv_heads`; past that vLLM replicates them, so per-GPU KV cost per token
stops falling. Llama-3.1-70B has 8 KV heads, so TP=16 doubles your GPU bill and
buys no additional KV sharding. Plans that assume "more GPUs means more context"
fail exactly here.

## Live monitoring

```bash
./kvmon.py --url http://localhost:8000 --predict --model llama-3.1-8b --gpu h100-sxm
```
```
  KV cache  [######################............]  65.2% total
    pinned  [============......................]  35.1% (rest is reclaimable prefix cache)
  requests   running    31   waiting     4   swapped    0
  tokens/s   prompt   4821.0   generated    912.4
  prefix     hit rate  62.9%   (610,816 / 971,776 tokens)
  preemptions    0 total   (0.0/min)
```

**Pinned utilisation is the number that predicts preemption**, and it is not the
number vLLM reports. With prefix caching on, a finished request's blocks stay
resident so a later request with the same prefix hits. They count as used but
are reclaimable on demand, so an idle engine with a warm cache reports ~90%
utilisation. Alerting on the raw gauge pages you on a healthy system; this tool
separates the two and alerts only on pinned blocks.

Works against vLLM, SGLang and the project 01 simulator, because it reads the
standard metric names.

## Alerts and what they mean

| Signal | Meaning | Action |
|---|---|---|
| pinned KV > 85% | a burst of long prompts will preempt | reduce `max_num_seqs`, or add capacity |
| `num_preemptions_total` rising | engine is discarding computed KV and redoing it | throughput falls while GPU util stays high |
| waiting grows, running flat | batch is capped | check whether the cap is `max_num_seqs` or memory |
| requests swapped > 0 | KV oversubscribed, PCIe now in the latency path | almost always worse than recompute |
| prefix hit rate falling | routing regression | see project 04 |
