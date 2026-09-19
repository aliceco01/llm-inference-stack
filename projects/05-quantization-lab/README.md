# 05 - Quantization Comparison Lab

> Quantization decisions need measured tradeoffs, not vibes.

Serves the same model under FP16/BF16, FP8, INT8 (W8A8), AWQ (W4A16) and
KV-only quantization, and measures VRAM, latency and **quality** together.

```bash
# rank candidates on a laptop before booking GPUs
./quantlab.py predict --model llama-3.1-70b --gpu h100-sxm --tp 2

# emit the serve command for each scheme
./quantlab.py plan --model meta-llama/Llama-3.1-8B-Instruct --gpu h100-sxm

# measured comparison against running endpoints
./quantlab.py run --config configs/llama8b-h100.yaml

# quality on its own
./quality.py --baseline http://localhost:8000 --candidate http://localhost:8003 \
             --model meta-llama/Llama-3.1-8B-Instruct --n 64
```

## The distinction that decides everything

**Weight-only (AWQ, GPTQ, W4A16)** stores weights at 4 bits and dequantizes
them inside the kernel before a normal fp16 matmul.

- Memory traffic drops ~4x, so memory-bound **decode gets much faster**.
- FLOPs are unchanged and dequant adds work, so compute-bound **prefill gets
  slower**.

**Weight and activation (FP8, INT8 W8A8)** quantizes both operands and uses
native low-precision tensor cores.

- Memory traffic halves **and** FLOPs double on supporting hardware.
- Both phases get faster.
- Activations contain outliers that weights do not, which is why INT8 requires
  SmoothQuant-style calibration and FP8 largely does not: its wider dynamic
  range absorbs them.

This is why a single "throughput" number for a quantization scheme is
meaningless. `quantlab.py run` therefore reports `decode_heavy` (128 in / 512
out) and `prefill_heavy` (4096 in / 32 out) separately. A blended number hides
the entire tradeoff, and the direction of the tradeoff depends on your traffic.

**KV cache quantization is a third, independent axis.** At long context the KV
cache dwarfs the weights, so `--kv-cache-dtype fp8` with full-precision weights
is often the highest value-per-risk change available. It is included as its own
scheme (`bf16-kv8`) rather than bundled into the others.

## Predict first, measure second

`predict` uses the roofline model in `llmkit.simulator.cost` plus the memory
model in `llmkit.kvcache` to rank schemes before you spend a GPU-hour. It
reports weights, KV cache, cacheable tokens, and separate decode/prefill
speedup estimates, along with the risks specific to each scheme, and it refuses
to imply anything about quality.

It also checks hardware support: FP8 needs compute capability 8.9 (Ada) or 9.0
(Hopper), so asking for FP8 on an A100 returns UNSUPPORTED rather than a
number.

## Measuring quality properly

`quality.py` runs three measurements against a live baseline and candidate:

**1. Reference agreement.** Greedy-decode identical prompts on both and compare
token by token. Reports exact-match rate and, more usefully, the **position of
first divergence**. Quantization damage compounds during generation, so a model
that diverges at token 3 is far worse than one that diverges at token 300 even
though both score 0% exact match.

**2. Logprob KL divergence.** Requests `top_logprobs` from both and computes
KL(baseline || candidate) per token. This is the sensitive measurement: it
detects degradation well before greedy output changes, because a token can
remain argmax while its margin collapses. Requires a backend that returns
`top_logprobs`; skipped cleanly if not.

**3. Task accuracy.** A small auto-scorable set covering the failure modes
quantization hits hardest: multi-step arithmetic, strict output formatting, and
instruction following.

### Why not perplexity

Perplexity on wikitext is the conventional quantization metric and it is close
to useless for a serving decision. It is dominated by common tokens, it is
insensitive to the long-generation degradation that breaks agent workloads, and
it routinely moves less than 1% for schemes that visibly damage multi-step
reasoning. It is reported in papers because it is cheap and comparable, not
because it predicts whether your users will notice.

## What the lab refuses to do

`run` will not present a latency win for a scheme whose quality was not
measured. The report's quality columns are blank for any unmeasured scheme and
the report says plainly that an unmeasured scheme is not a candidate. The entire
failure mode this project exists to prevent is shipping a 2x throughput win that
quietly costs 8 points of task accuracy.

## Checkpoint caveat

4-bit schemes need a **pre-quantized checkpoint**. Passing `--quantization awq`
to an fp16 repo does not produce AWQ weights; it fails or silently misbehaves.
Either point at an already-quantized repo (as `configs/llama8b-h100.yaml` does)
or quantize offline with llm-compressor or AutoAWQ. FP8 is the exception: vLLM
can quantize weights to FP8 at load time, which is a large part of why it is the
sensible default on Hopper.
