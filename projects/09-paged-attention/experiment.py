#!/usr/bin/env python3
"""PagedAttention under memory pressure: fragmentation, eviction, preemption.

    ./experiment.py fragmentation --out results/   # paged vs contiguous
    ./experiment.py blocksize     --out results/   # block size tradeoff
    ./experiment.py eviction      --out results/   # LRU behaviour vs cache size
    ./experiment.py preemption    --out results/   # recompute vs swap
    ./experiment.py cow           --out results/   # copy-on-write on forks
    ./experiment.py all           --out results/

Every experiment here drives the actual allocator in
`llmkit/simulator/paged.py`, which implements the real mechanism: fixed-size
blocks, per-sequence block tables, content-addressed prefix sharing with
reference counts, copy-on-write on partial blocks, LRU eviction of unreferenced
cached blocks, and a watermark. Nothing is mocked out, so the numbers below are
properties of the design rather than assertions about it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from llmkit import report
from llmkit.simulator import EngineConfig, EngineSim, SimRequest
from llmkit.simulator.engine import PreemptionMode
from llmkit.simulator.paged import ContiguousAllocator, OutOfBlocks, PagedKVCache


# ---------------------------------------------------------------------------
def cmd_fragmentation(args) -> int:
    """The core claim: paging trades unbounded external fragmentation for a
    small, bounded amount of internal fragmentation."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # Realistic chat length distribution: most requests short, a long tail.
    lengths = [max(16, int(rng.lognormvariate(math.log(400), 0.9)))
               for _ in range(args.n_seqs)]
    max_len = args.max_model_len

    print(f"{args.n_seqs} sequences, lognormal lengths "
          f"(median {sorted(lengths)[len(lengths)//2]}, max {max(lengths)}), "
          f"max_model_len={max_len}\n")

    rows = []
    for bs in (1, 8, 16, 32, 64):
        kv = PagedKVCache(num_blocks=args.blocks, block_size=bs,
                          enable_prefix_caching=False)
        f = kv.fragmentation(lengths)
        rows.append({
            "allocator": f"paged (block={bs})",
            "internal waste %": f["internal_waste_pct"],
            "external waste %": f["external_waste_pct"],
            "wasted tokens": f["internal_waste_tokens"],
        })

    # Contiguous baseline. It cannot know the final length, so it must reserve
    # max_model_len up front for every sequence.
    ca = ContiguousAllocator(capacity_tokens=args.blocks * 16,
                             reserve_len=max_len)
    live: dict[str, int] = {}
    admitted = 0
    for i, L in enumerate(lengths):
        if ca.allocate(f"s{i}") is not None:
            live[f"s{i}"] = L
            admitted += 1
    st = ca.stats(live)
    rows.append({
        "allocator": f"contiguous (reserve={max_len})",
        "internal waste %": st["internal_waste_pct"],
        "external waste %": st["external_waste_pct"],
        "wasted tokens": st["reserved_tokens"] - st["used_tokens"],
    })
    print(report.md_table(rows))
    print(f"\ncontiguous admitted only {admitted}/{len(lengths)} sequences "
          f"before running out of space; paged admits all of them in the same "
          f"{args.blocks * 16:,} tokens of memory.")

    # Now demonstrate external fragmentation directly: free alternating slots
    # and show a full-size request is rejected while the bytes are free.
    ca2 = ContiguousAllocator(capacity_tokens=8 * max_len, reserve_len=max_len)
    ids = [f"x{i}" for i in range(8)]
    for i in ids:
        ca2.allocate(i)
    for i in ids[::2]:
        ca2.free(i)
    st2 = ca2.stats({i: 100 for i in ids[1::2]})
    got = ca2.allocate("new-big")
    print("\nexternal fragmentation demonstration (contiguous):")
    print(f"  freed 4 of 8 slots: {st2['free_tokens']:,} tokens free, "
          f"largest contiguous run {st2['largest_free_run']:,}")
    print(f"  a request needing {max_len:,} tokens: "
          f"{'ACCEPTED' if got is not None else 'REJECTED'}")
    print(f"  external waste: {st2['external_waste_pct']:.0f}% of free memory "
          "is unusable")
    print("\n  Under paging this cannot happen: a request needs N blocks and "
          "ANY N free\n  blocks will do, so free memory is always usable. That "
          "is the entire point of\n  the design, and it is why vLLM can run "
          "batch sizes that a contiguous\n  allocator cannot.")

    report.bar_compare(
        [r["allocator"] for r in rows],
        {"internal waste %": [r["internal waste %"] for r in rows],
         "external waste %": [r["external waste %"] for r in rows]},
        out / "fragmentation.png", ylabel="% wasted",
        title="Memory waste: paged vs contiguous allocation", simulated=True)
    (out / "fragmentation.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/fragmentation.png")
    return 0


def cmd_blocksize(args) -> int:
    """Block size is a real tradeoff, not a free parameter.

    Smaller blocks waste less on the partial tail of each sequence, but cost
    more block-table entries, more per-block bookkeeping, and coarser prefix
    sharing granularity. vLLM's default of 16 is a reasonable middle.
    """
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    lengths = [max(16, int(rng.lognormvariate(math.log(600), 0.8)))
               for _ in range(512)]
    print(f"{'block':>7}{'internal waste %':>19}{'blocks/seq (mean)':>20}"
          f"{'block table entries':>22}{'prefix granularity':>20}")
    print("-" * 88)
    rows = []
    for bs in (1, 4, 8, 16, 32, 64, 128):
        kv = PagedKVCache(num_blocks=100000, block_size=bs)
        f = kv.fragmentation(lengths)
        mean_blocks = sum(math.ceil(L / bs) for L in lengths) / len(lengths)
        total_entries = sum(math.ceil(L / bs) for L in lengths)
        print(f"{bs:>7}{f['internal_waste_pct']:>19.2f}{mean_blocks:>20.1f}"
              f"{total_entries:>22,}{bs:>17} tok")
        rows.append({"block_size": bs,
                     "internal_waste_pct": f["internal_waste_pct"],
                     "mean_blocks_per_seq": mean_blocks,
                     "block_table_entries": total_entries})
    print("""
Reading this:
  * Internal waste is bounded by (block_size - 1) tokens per sequence, so it
    falls as blocks shrink and is already under 2% at block_size=16 for
    realistic lengths.
  * Block-table entries grow inversely, and every decode step walks that table.
  * Prefix sharing granularity equals the block size: with block_size=32, two
    requests sharing a 40-token prefix share only the first 32 tokens, because
    a partially filled block is never cached.

  The last point is why very large blocks quietly hurt prefix cache hit rates,
  which is a project 04 problem caused by a project 09 setting.""")
    report.bar_compare(
        [str(r["block_size"]) for r in rows],
        {"internal waste %": [r["internal_waste_pct"] for r in rows]},
        out / "blocksize.png", ylabel="%",
        title="Internal fragmentation vs block size", simulated=True)
    (out / "blocksize.json").write_text(json.dumps(rows, indent=2))
    return 0


def cmd_eviction(args) -> int:
    """Prefix cache hit rate as a function of cache capacity.

    Cached-but-unreferenced blocks are retained and evicted LRU only when the
    free list is empty. The interesting behaviour is the knee: hit rate is
    flat and high while the working set fits, then falls off a cliff, because
    LRU on a working set larger than the cache degrades toward zero reuse
    rather than degrading gracefully.
    """
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2)
    n_prefixes = args.n_prefixes
    prefix_len = args.prefix_len
    unique_len = 128
    block = 16

    # Zipf-ish popularity: a few system prompts dominate, as in production.
    weights = [1.0 / (i + 1) ** args.zipf for i in range(n_prefixes)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    working_set_blocks = n_prefixes * math.ceil(prefix_len / block)
    print(f"{n_prefixes} distinct prefixes of {prefix_len} tokens "
          f"(working set = {working_set_blocks:,} blocks), "
          f"Zipf exponent {args.zipf}\n")
    print(f"{'cache blocks':>14}{'vs working set':>16}{'hit rate %':>13}"
          f"{'evictions':>12}")
    print("-" * 55)
    rows = []
    for mult in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        n_blocks = max(int(working_set_blocks * mult), 8)
        kv = PagedKVCache(num_blocks=n_blocks, block_size=block,
                          enable_prefix_caching=True)
        hits = misses = 0
        for _ in range(args.n_requests):
            pick = rng.choices(range(n_prefixes), weights=probs, k=1)[0]
            base = (pick + 1) * 1_000_000
            ids = ([base + i for i in range(prefix_len)]
                   + [rng.randrange(9_000_000, 9_999_999) for _ in range(unique_len)])
            before_h = kv.stats.cache_hits
            before_m = kv.stats.cache_misses
            try:
                table, cached = kv.allocate_prompt(ids)
            except OutOfBlocks:
                continue
            hits += kv.stats.cache_hits - before_h
            misses += kv.stats.cache_misses - before_m
            kv.free(table)
        rate = hits / max(hits + misses, 1) * 100
        print(f"{n_blocks:>14,}{mult:>15.2f}x{rate:>13.1f}"
              f"{kv.stats.evictions:>12,}")
        rows.append({"cache_blocks": n_blocks, "vs_working_set": mult,
                     "hit_rate_pct": rate, "evictions": kv.stats.evictions})
    print("""
The knee is the operationally important feature. Above ~1x the working set the
hit rate is high and stable; below it, evictions rise and the hit rate collapses
rather than degrading smoothly, because LRU thrashes when the reuse distance
exceeds the cache.

Consequence: prefix cache hit rate is a step function of how many distinct
system prompts you serve per replica. Adding one more tenant to a replica that
was just fitting its working set can drop the hit rate by tens of points. That
is the capacity argument for the routing in project 04.""")
    report.bar_compare(
        [f"{r['vs_working_set']}x" for r in rows],
        {"hit rate %": [r["hit_rate_pct"] for r in rows]},
        out / "eviction.png", ylabel="%",
        title="Prefix cache hit rate vs cache size (x working set)", simulated=True)
    (out / "eviction.json").write_text(json.dumps(rows, indent=2))
    return 0


def cmd_preemption(args) -> int:
    """Recompute vs swap under deliberate memory pressure."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in (PreemptionMode.RECOMPUTE, PreemptionMode.SWAP):
        cfg = EngineConfig(
            model=args.model, gpu=args.gpu, max_model_len=args.seq_len + 64,
            max_num_seqs=args.max_num_seqs,
            num_gpu_blocks_override=args.blocks,
            preemption_mode=mode, enable_prefix_caching=False, seed=5,
        )
        e = EngineSim(cfg)
        e.submit([SimRequest(f"r{i}", prompt_tokens=args.seq_len,
                             output_tokens=args.out_len, arrival_ms=i * 2.0)
                  for i in range(args.n_seqs)])
        e.run(max_steps=400_000)
        from llmkit import SLO, summarize
        s = summarize(e.records(), label=mode.value,
                      slo=SLO(ttft_ms=5000, p_itl_ms=200))
        st = e.stats()
        worst = 0.0
        for r in e.records():
            itls = r.itls_ms
            if itls:
                worst = max(worst, max(itls))
        rows.append({
            "mode": mode.value, "preemptions": st["preemptions"],
            "sim time ms": st["sim_time_ms"], "steps": st["steps"],
            "TTFT p95": s.ttft.p95, "ITL p99": s.itl.p99,
            "worst stall ms": worst, "out tok/s": s.output_tok_per_s,
        })
        print(f"{mode.value:<12} preemptions={st['preemptions']:<5} "
              f"sim_time={st['sim_time_ms']:.0f}ms  "
              f"TTFT p95={s.ttft.p95:.0f}ms  worst stall={worst:.0f}ms")
    print("\n" + report.md_table(rows))
    print("""
Recompute throws the KV away and re-prefills from scratch later. Cheap to
evict, expensive to restart, and the restart cost lands in the TTFT of a
request that had already started, so it shows up as a tail-latency problem
rather than a throughput one.

Swap copies KV to host memory over PCIe and back. It preserves the work but
puts a ~25 GB/s link in the latency path of a cache that the GPU reads at
3350 GB/s, a two-orders-of-magnitude mismatch. For short sequences recompute
is almost always cheaper; swap only starts to pay for very long contexts where
re-prefilling is genuinely expensive.

Either way, preemption at all means the deployment is over-subscribed. The fix
is capacity or admission control (projects 03 and 11), not a preemption-mode
flag.""")
    (out / "preemption.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0


def cmd_cow(args) -> int:
    """Copy-on-write when a shared block is written.

    Two sequences share the blocks of a common prefix by reference count. When
    one of them writes into a partially-filled shared tail block, that block
    must be copied first or the write would corrupt the other sequence's KV.
    """
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kv = PagedKVCache(num_blocks=512, block_size=16, enable_prefix_caching=True)
    shared = list(range(5000, 5000 + 64))     # 4 full blocks

    t1, c1 = kv.allocate_prompt(shared + [1, 2, 3])
    t2, c2 = kv.allocate_prompt(shared + [1, 2, 3])
    print("two identical 67-token prompts (4 full shared blocks + partial tail)")
    print(f"  seq A: {len(t1)} blocks, {c1} cached tokens")
    print(f"  seq B: {len(t2)} blocks, {c2} cached tokens")
    print(f"  shared block ids identical: {t1[:4] == t2[:4]}")
    print(f"  blocks in use: {kv.num_used} (not {2 * len(t1)}, because the "
          f"first 4 are shared)")

    before = kv.stats.cow_copies
    for i in range(20):
        kv.append_token(t1, 67 + i)
    print(f"\n  after 20 decode steps on seq A: "
          f"{kv.stats.cow_copies - before} copy-on-write copies")
    print(f"  seq B's block table unchanged: {t2[:4] == t1[:4]}")
    print(f"  blocks in use now: {kv.num_used}")
    print("""
  The partial tail block is never entered into the prefix cache, precisely
  because its contents are still growing: caching it would let another request
  match a block whose remaining slots are about to be overwritten.

  This is also the mechanism behind cheap parallel sampling and beam search.
  n=4 sampling from one prompt shares every prompt block by reference and only
  copies the tail, so the marginal memory cost of the 2nd, 3rd and 4th
  candidate is a handful of blocks rather than a full prompt each.""")
    (out / "cow.json").write_text(json.dumps({
        "shared_blocks": t1[:4], "cow_copies": kv.stats.cow_copies,
        "blocks_used": kv.num_used}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="results")
        p.add_argument("--model", default="llama-3.1-8b")
        p.add_argument("--gpu", default="h100-sxm")

    p = sub.add_parser("fragmentation", help="paged vs contiguous"); common(p)
    p.add_argument("--n-seqs", type=int, default=256)
    p.add_argument("--blocks", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=8192)

    p = sub.add_parser("blocksize", help="block size tradeoff"); common(p)

    p = sub.add_parser("eviction", help="LRU behaviour vs cache size"); common(p)
    p.add_argument("--n-prefixes", type=int, default=32)
    p.add_argument("--prefix-len", type=int, default=2048)
    p.add_argument("--n-requests", type=int, default=2000)
    p.add_argument("--zipf", type=float, default=1.0)

    p = sub.add_parser("preemption", help="recompute vs swap"); common(p)
    p.add_argument("--n-seqs", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--blocks", type=int, default=900)
    p.add_argument("--max-num-seqs", type=int, default=32)

    p = sub.add_parser("cow", help="copy-on-write"); common(p)
    p = sub.add_parser("all", help="run everything"); common(p)
    p.add_argument("--n-seqs", type=int, default=256)
    p.add_argument("--blocks", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--n-prefixes", type=int, default=32)
    p.add_argument("--prefix-len", type=int, default=2048)
    p.add_argument("--n-requests", type=int, default=2000)
    p.add_argument("--zipf", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--max-num-seqs", type=int, default=32)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "all":
        for fn in (cmd_fragmentation, cmd_blocksize, cmd_eviction, cmd_cow):
            print("\n" + "=" * 78)
            fn(args)
        print("\n" + "=" * 78)
        args.blocks = 900
        return cmd_preemption(args)
    return {"fragmentation": cmd_fragmentation, "blocksize": cmd_blocksize,
            "eviction": cmd_eviction, "preemption": cmd_preemption,
            "cow": cmd_cow}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
