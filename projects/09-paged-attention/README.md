# 09 - PagedAttention Deep-Dive

> Understanding the scheduler beats memorizing flags.

A working implementation of the PagedAttention block allocator, driven under
memory pressure to document fragmentation, eviction and preemption behaviour.

```bash
./experiment.py fragmentation   # paged vs contiguous
./experiment.py blocksize       # the block size tradeoff
./experiment.py eviction        # LRU hit rate vs cache size
./experiment.py preemption      # recompute vs swap
./experiment.py cow             # copy-on-write on shared blocks
./experiment.py all --out results/
```

The allocator lives in [`llmkit/simulator/paged.py`](../../packages/llmkit/llmkit/simulator/paged.py)
and is the real mechanism, not a description of it: fixed-size blocks,
per-sequence block tables, content-addressed prefix sharing with reference
counts, copy-on-write on partial blocks, LRU eviction of unreferenced cached
blocks, and an allocation watermark. A `ContiguousAllocator` sits beside it as
the pre-paging baseline.

## The claim, and the measurement

The argument for PagedAttention is usually stated as "it reduces memory waste".
That is true but imprecise. What it actually does is **trade unbounded external
fragmentation for a small, bounded amount of internal fragmentation.**

Driving both allocators over the same sequence lengths:

```
paged, block_size=16   internal waste   4.2%     external waste   0%
contiguous, reserve 4k internal waste  97.7%     external waste  50%
```

(from an early run of the allocator on four sequences of 100/1000/17/33 tokens
against a 4096-token reservation; `./experiment.py fragmentation` runs the full
lognormal distribution.)

Two separate failures in the contiguous case:

**Over-reservation.** A contiguous allocator cannot know a sequence's final
length, so it must reserve `max_model_len` up front. A 100-token chat reserves
4096, or 128k on a long-context model. That is the 97.7%.

**External fragmentation.** Free alternating slots and you have plenty of free
memory in pieces too small to serve a full-size request. The experiment
demonstrates this directly: 4 of 8 slots freed, half of memory available, and a
full-size request still **rejected**.

Under paging the second failure is structurally impossible. A request needs N
blocks and *any* N free blocks will do, so free memory is always usable. That is
the whole design, and it is why vLLM sustains batch sizes a contiguous allocator
cannot reach on identical hardware.

## Block size is a real tradeoff

`./experiment.py blocksize` sweeps it. Three things move in different
directions:

- **Internal waste** is bounded by `block_size - 1` tokens per sequence, so it
  falls as blocks shrink. At block_size=16 it is already under 2% for realistic
  lengths.
- **Block-table entries** grow inversely, and every decode step walks that
  table.
- **Prefix sharing granularity equals the block size.** Two requests sharing a
  40-token prefix share only 32 tokens at block_size=32, because a partially
  filled block is never entered into the cache.

That last point is the non-obvious one: large blocks quietly depress prefix
cache hit rates. It is a project 04 symptom with a project 09 cause.

## Eviction has a knee, not a slope

`./experiment.py eviction` varies cache capacity against a Zipf-distributed set
of system prompts and reports hit rate.

Hit rate is high and flat while the cache exceeds the working set, then
**collapses** below it rather than degrading gracefully, because LRU thrashes
once reuse distance exceeds capacity.

The operational consequence: prefix cache hit rate is close to a step function
of how many distinct system prompts a replica serves. Adding one tenant to a
replica that was just fitting its working set can cost tens of points of hit
rate, and the replica will look fine on every other metric. This is the capacity
argument underneath project 04's routing.

## Preemption: recompute vs swap

When the engine runs out of blocks it does not crash, it **preempts**. This is
the single most misdiagnosed behaviour in vLLM operations, because the symptom
is a latency cliff with high GPU utilisation, which looks like "the model got
slower" rather than "we ran out of memory".

**Recompute** frees the blocks and re-prefills from scratch later. Cheap to
evict, expensive to restart, and the restart cost lands in the TTFT of a request
that had already started, so it surfaces as tail latency rather than reduced
throughput.

**Swap** copies KV to host memory over PCIe and back. It preserves the work but
inserts a ~25 GB/s link into the latency path of a cache the GPU reads at
3350 GB/s. For short sequences recompute almost always wins; swap only starts to
pay at very long contexts where re-prefilling is genuinely expensive.

The victim is chosen LIFO (newest first), matching vLLM: the oldest requests are
closest to finishing, so evicting them wastes the most completed work and does
the most tail-latency damage.

Either mode means the deployment is over-subscribed. The fix is capacity or
admission control (projects 03 and 11), not a preemption-mode flag.

## Copy-on-write and why parallel sampling is cheap

`./experiment.py cow` shows two identical prompts sharing four full blocks by
reference count, then one of them decoding. The shared tail block is copied on
first write, the other sequence's block table is untouched, and the partial tail
is never cached (its remaining slots are about to be overwritten, so caching it
would let another request match a block that is still changing).

This is also the mechanism behind cheap `n>1` sampling and beam search: four
candidates from one prompt share every prompt block by reference and copy only
their tails, so the marginal cost of candidates 2, 3 and 4 is a handful of
blocks rather than a full prompt each.

## Reproducing this against real vLLM

The behaviours above are observable on a real engine through its metrics:

| Behaviour | Metric to watch |
|---|---|
| preemption | `vllm:num_preemptions_total` |
| cache pressure | `vllm:gpu_cache_usage_perc` (see project 03 on pinned vs reclaimable) |
| eviction / hit rate | `vllm:prefix_cache_hits_total` over `vllm:prefix_cache_queries_total` |
| swap | `vllm:num_requests_swapped` |

Drive load with project 02, watch with project 03's `kvmon.py`, and shrink the
cache with `--num-gpu-blocks-override` to reach the interesting regime without
needing a large workload.
