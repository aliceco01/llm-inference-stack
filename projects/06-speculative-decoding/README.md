# 06 - Speculative Decoding Pipeline

> Faster decode with zero quality loss is the closest thing to a free lunch in
> inference.

A draft model proposes k tokens, the target verifies all of them in one forward
pass, and rejection sampling accepts a prefix. Includes acceptance-rate
tracking, because acceptance is the number that decides whether any of it pays.

```bash
./specdec.py plan --batch-size 1          # which strategy, what k
./specdec.py plan --batch-size 64         # watch the gain collapse
./specdec.py curve --draft-cost 0.16      # sensitivity to acceptance rate
./specdec.py simulate --alpha 0.72 --k 4  # end to end, no GPU
./specdec.py watch --url http://localhost:8000   # live acceptance tracking
./specdec.py serve-cmd --target meta-llama/Llama-3.1-70B-Instruct
```

## Why it is lossless

The acceptance test is constructed so that accepted tokens are distributed
*exactly* as the target model would have produced them. Accept draft token x
with probability min(1, p_target(x) / p_draft(x)); on rejection, sample from the
normalised residual distribution max(0, p_target - p_draft). The output
distribution is provably identical to plain target sampling.

So this is not an approximation with a quality/speed knob. It is strictly a
speed optimisation, which is what makes it unusual and worth the complexity.

## Why it works at all

Decode is memory-bandwidth bound: every step reads the full weight matrix from
HBM to produce one token. Verifying k+1 tokens in one forward reads those
weights **once**, so it costs barely more than producing a single token. You
are spending idle FLOPs to buy back memory traffic.

This also tells you exactly when it stops working, which is the part most
write-ups omit: see "the concurrency cliff" below.

## The arithmetic that gets misreported

Expected accepted draft tokens is **not** `k * alpha`. Rejection sampling stops
at the first rejection, so:

```
E[accepted] = alpha + alpha^2 + ... + alpha^k
E[tokens per iteration] = (1 - alpha^(k+1)) / (1 - alpha)      (includes the bonus token)
speedup = E[tokens] / (k * draft_cost_ratio + 1 + verify_overhead * k)
```

At `alpha = 0.7, k = 8`, the naive count claims 5.6 accepted draft tokens. The
true expectation is **2.20**, giving 3.20 tokens per iteration rather than 6.6.
The error grows with k exactly where the naive model promises the largest wins.

`./specdec.py curve` prints this comparison directly, along with the breakeven
acceptance rate per k: below that rate you are paying for draft forwards that
get thrown away, and speculation is slower than not speculating.

There is always an optimal k, because gains decay geometrically (`alpha^k`)
while cost grows linearly (`k * c`).

## The concurrency cliff

Speculation trades FLOPs for bandwidth. That trade only pays while decode is
bandwidth-bound. At high batch sizes the target forward is already
compute-bound, verifying k+1 tokens per sequence costs real time, and the
speedup collapses toward 1.0 or below.

`./specdec.py plan --batch-size 64` models this and will tell you there is no
useful gain. **Speculative decoding is a low-concurrency optimisation.** It is
excellent for single-user interactive latency and for lightly loaded endpoints,
and it can make a saturated server slower. Benchmarks that report a 2.5x speedup
at batch size 1 and omit the concurrency sweep are not wrong, they are just not
describing production.

## Draft strategies

| Strategy | Cost ratio | Extra VRAM | Notes |
|---|---|---|---|
| Draft model (1B for a 70B target) | ~0.16 | ~2.5 GiB | Must share the target's tokenizer. A different family cannot be used at all. |
| N-gram / prompt lookup | ~0.005 | 0 | No model. Copies continuations from the prompt. |
| EAGLE-2 head | ~0.06 | ~1.2 GiB | Highest acceptance per cost; needs a head trained for that exact checkpoint. |
| Medusa heads | ~0.04 | ~1.5 GiB | Independent heads, so later positions are weak. |

**Try n-gram first.** It needs no checkpoint, no extra VRAM, and no training,
and on summarisation, RAG, code editing and any task that quotes its input it is
often competitive with a real draft model. It is useless for open-ended
generation, which is precisely the kind of thing you should measure rather than
assume.

Note the VRAM column interacts with project 03: a 2.5 GiB draft model is 2.5 GiB
that is no longer KV cache, which lowers max concurrency. On a memory-tight
deployment the draft model can cost more in lost batch size than it gains in
decode speed.

## Acceptance rate is a production metric, not a constant

`./specdec.py watch` tracks `vllm:spec_decode_*` counters and computes windowed
and cumulative acceptance. This matters because acceptance depends on the
traffic mix: prompt-lookup speculation is excellent on summarisation and
near-useless on open-ended chat. A configuration that was a 2x win at launch
becomes a net loss when traffic shifts, and nothing will alert you unless you
are watching this number.

The tool warns when p5 acceptance drops below the breakeven rate for your
configured k, which means speculation is already losing on part of your traffic.

## Simulation

`./specdec.py simulate` runs the full engine simulator with and without
speculation and reports the ITL difference. The simulator samples acceptance
with the correct truncated-geometric behaviour (stop at first rejection), not
`k * alpha`. Its output is labelled simulated, because it is.
