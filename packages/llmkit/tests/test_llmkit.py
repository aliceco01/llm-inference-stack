"""Tests for the shared core.

These cover the properties the whole stack depends on, in particular the ones
that were actually wrong at some point during development:

  * timestamp sentinels (0 is a legal virtual-clock value, so "unset" must be -1)
  * TTFT measured to first CONTENT token, not first frame
  * ITL amortised across multi-token frames
  * Little's law consistency in the metrics layer
  * KV math against known values (GQA, MLA, sliding window)
  * prefix cache sharing, eviction and copy-on-write
  * speculative decoding's truncated-geometric acceptance

    pytest packages/llmkit/tests -v
"""

from __future__ import annotations

import math

import pytest

from llmkit import (
    SLO,
    RequestRecord,
    ServingConfig,
    TokenEvent,
    compute_budget,
    get_gpu,
    get_model,
    percentile,
    summarize,
)
from llmkit.types import UNSET


# ---------------------------------------------------------------------------
# Record semantics
# ---------------------------------------------------------------------------
def test_unset_timestamps_are_negative_not_zero():
    """0 is a legal timestamp on the simulator's virtual clock.

    Using falsiness to mean "unset" silently NaN'd every simulated metric.
    """
    r = RequestRecord("x")
    assert r.t_send_ns == UNSET < 0
    assert math.isnan(r.ttft_ms) and math.isnan(r.e2e_ms)

    r2 = RequestRecord("y", t_send_ns=0, t_first_token_ns=10_000_000,
                       t_done_ns=20_000_000, output_tokens=2)
    assert r2.ttft_ms == 10.0
    assert r2.e2e_ms == 20.0


def test_ttft_uses_first_content_token_not_first_frame():
    r = RequestRecord("x", t_send_ns=0,
                      t_first_chunk_ns=5_000_000,      # role-only delta
                      t_first_token_ns=30_000_000,     # actual content
                      t_done_ns=40_000_000)
    assert r.ttft_first_chunk_ms == 5.0
    assert r.ttft_ms == 30.0
    assert r.ttft_ms > r.ttft_first_chunk_ms


def test_itl_amortises_multi_token_frames():
    """A frame carrying k tokens must not look like one long ITL."""
    r = RequestRecord("x", t_send_ns=0)
    r.token_events = [
        TokenEvent(0),
        TokenEvent(10_000_000, content_tokens=1),
        TokenEvent(30_000_000, content_tokens=4),   # 20ms gap, 4 tokens
    ]
    itls = r.itls_ms
    assert itls[0] == 10.0
    assert itls[1:] == [5.0, 5.0, 5.0, 5.0]


def test_tpot_requires_two_tokens():
    r = RequestRecord("x", t_send_ns=0, t_first_token_ns=1_000_000,
                      t_done_ns=2_000_000, output_tokens=1)
    assert math.isnan(r.tpot_ms)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_percentile_nearest_rank_returns_observed_value():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == 5      # an actual observation
    assert percentile(xs, 95) == 10
    assert percentile(xs, 50, "linear") == 5.5   # interpolated, for contrast


def test_percentile_handles_empty_and_nan():
    assert math.isnan(percentile([], 50))
    assert percentile([1.0, float("nan"), 3.0], 50) in (1.0, 3.0)


def _synthetic_run(n=20, concurrency=4, e2e_ms=100, out_tokens=9):
    recs = []
    for i in range(n):
        start = (i // concurrency) * e2e_ms * 1_000_000
        r = RequestRecord(f"r{i}", t_submit_ns=start, t_send_ns=start,
                          t_first_token_ns=start + 20_000_000,
                          t_last_token_ns=start + e2e_ms * 1_000_000,
                          t_done_ns=start + e2e_ms * 1_000_000,
                          prompt_tokens=50, output_tokens=out_tokens)
        r.token_events = [TokenEvent(start + 20_000_000 + k * 10_000_000)
                          for k in range(out_tokens)]
        recs.append(r)
    return recs


def test_littles_law_holds_on_synthetic_run():
    """N = X * R must hold, or the harness is measuring something else."""
    s = summarize(_synthetic_run(), label="t", concurrency=4,
                  slo=SLO(ttft_ms=100, p_itl_ms=50))
    assert s.achieved_concurrency == pytest.approx(4.0, abs=0.1)
    assert s.littles_law_error == pytest.approx(0.0, abs=0.02)
    assert not s.warnings


def test_throughput_uses_window_not_sum_of_rates():
    s = summarize(_synthetic_run(n=20, concurrency=4, e2e_ms=100, out_tokens=9),
                  label="t", concurrency=4)
    # 20 requests x 9 tokens over a 500ms window = 360 tok/s.
    assert s.output_tok_per_s == pytest.approx(360.0, rel=0.02)


def test_goodput_excludes_slo_violations():
    recs = _synthetic_run()
    strict = summarize(recs, label="t", concurrency=4,
                       slo=SLO(ttft_ms=1.0, p_itl_ms=50))
    assert strict.goodput_ratio == 0.0
    loose = summarize(recs, label="t", concurrency=4,
                      slo=SLO(ttft_ms=1000, p_itl_ms=1000))
    assert loose.goodput_ratio == 1.0


def test_warmup_exclusion_is_reported():
    s = summarize(_synthetic_run(n=20), label="t", concurrency=4,
                  warmup_requests=8)
    assert s.n_excluded_warmup == 8
    assert s.n_ok == 12


# ---------------------------------------------------------------------------
# KV cache math
# ---------------------------------------------------------------------------
def test_kv_bytes_per_token_matches_known_values():
    """2 * layers * kv_heads * head_dim * dtype_bytes."""
    m = get_model("llama-3.1-8b")
    assert m.kv_bytes_per_token("fp16") == 2 * 32 * 8 * 128 * 2
    assert m.kv_bytes_per_token("fp16") / 1024 == 128.0          # KiB
    assert m.kv_bytes_per_token("fp8") == m.kv_bytes_per_token("fp16") / 2

    m70 = get_model("llama-3.1-70b")
    assert m70.kv_bytes_per_token("fp16") / 1024 == 320.0


def test_gqa_ratio_is_the_common_error():
    """Sizing off attention heads instead of KV heads overestimates by the ratio."""
    m = get_model("llama-3.1-70b")
    assert m.gqa_ratio == 8.0
    naive = 2 * m.num_layers * m.num_attention_heads * m.head_dim * 2
    assert naive == m.kv_bytes_per_token("fp16") * 8


def test_mla_has_no_factor_of_two():
    """DeepSeek-V3's compressed cache is smaller per token than an 8B GQA model."""
    d = get_model("deepseek-v3")
    assert d.attn_kind == "mla"
    expected = d.num_layers * (d.kv_lora_rank + d.qk_rope_head_dim) * 2
    assert d.kv_bytes_per_token("fp16") == expected
    assert d.kv_bytes_per_token("fp16") < get_model("llama-3.1-8b").kv_bytes_per_token("fp16")


def test_sliding_window_caps_kv_growth():
    g = get_model("gemma-2-9b")
    assert g.sliding_window == 4096
    naive = g.kv_bytes_per_token("fp16") * 32768
    assert g.kv_bytes_for_sequence(32768) < naive


def test_parameter_estimates_are_close_to_published():
    for name, tol in [("llama-3.1-8b", 0.02), ("llama-3.1-70b", 0.02),
                      ("qwen2.5-7b", 0.02), ("mixtral-8x7b", 0.03),
                      ("gemma-2-9b", 0.03), ("deepseek-v3", 0.06)]:
        m = get_model(name)
        err = abs(m.estimate_params() / 1e9 - m.params_b) / m.params_b
        assert err < tol, f"{name}: estimate off by {err*100:.1f}%"


def test_moe_active_params_much_less_than_total():
    mx = get_model("mixtral-8x7b")
    assert mx.active_params() / 1e9 < 20      # ~13B active of 46.7B total
    assert mx.active_params() < mx.params_b * 1e9


def test_budget_flags_impossible_configuration():
    b = compute_budget("llama-3.1-70b", "h100-sxm", ServingConfig(tp=1))
    assert not b.fits
    assert any("weights alone" in p for p in b.problems)


def test_tp_stops_sharding_kv_past_kv_heads():
    m = get_model("llama-3.1-70b")   # 8 KV heads
    b8 = compute_budget(m, "h100-sxm", ServingConfig(tp=8))
    b16 = compute_budget(m, "h100-sxm", ServingConfig(tp=16))
    assert not b8.kv_heads_replicated
    assert b16.kv_heads_replicated
    assert b16.kv_bytes_per_token == b8.kv_bytes_per_token


def test_h100_8b_budget_is_in_the_expected_range():
    """Sanity check against what real vLLM reports (~400k KV tokens)."""
    b = compute_budget("llama-3.1-8b", "h100-sxm",
                       ServingConfig(max_model_len=8192))
    assert 350_000 < b.max_cached_tokens < 500_000
    assert 40 < b.kv_cache_gib < 56


# ---------------------------------------------------------------------------
# Paged allocator
# ---------------------------------------------------------------------------
def test_prefix_blocks_are_shared_by_reference():
    from llmkit.simulator.paged import PagedKVCache
    kv = PagedKVCache(num_blocks=128, block_size=16)
    shared = list(range(1000, 1064))          # 4 full blocks
    t1, c1 = kv.allocate_prompt(shared + [1, 2, 3])
    t2, c2 = kv.allocate_prompt(shared + [9, 9, 9])
    assert c1 == 0 and c2 == 64
    assert t1[:4] == t2[:4]
    assert kv.num_used < 2 * len(t1)


def test_distinct_prompts_do_not_share():
    from llmkit.simulator.paged import PagedKVCache
    kv = PagedKVCache(num_blocks=128, block_size=16)
    _, c1 = kv.allocate_prompt(list(range(0, 64)))
    _, c2 = kv.allocate_prompt(list(range(500, 564)))
    assert c1 == 0 and c2 == 0


def test_cached_blocks_survive_free_and_are_evictable():
    from llmkit.simulator.paged import PagedKVCache
    kv = PagedKVCache(num_blocks=128, block_size=16)
    ids = list(range(2000, 2064))
    t, _ = kv.allocate_prompt(ids)
    kv.free(t)
    assert kv.num_evictable > 0
    _, cached = kv.allocate_prompt(ids)
    assert cached == 64            # cold cache hit after free


def test_copy_on_write_protects_shared_tail():
    from llmkit.simulator.paged import PagedKVCache
    kv = PagedKVCache(num_blocks=256, block_size=16)
    ids = list(range(3000, 3067))     # 4 full blocks + 3-token tail
    t1, _ = kv.allocate_prompt(ids)
    t2, _ = kv.allocate_prompt(ids)
    before = list(t2)
    for i in range(20):
        kv.append_token(t1, 67 + i)
    assert t2 == before               # the other sequence is untouched


def test_paging_has_no_external_fragmentation():
    from llmkit.simulator.paged import PagedKVCache
    kv = PagedKVCache(num_blocks=1024, block_size=16)
    f = kv.fragmentation([100, 1000, 17, 33])
    assert f["external_waste_pct"] == 0.0
    assert f["internal_waste_pct"] < 10.0


def test_contiguous_allocator_suffers_external_fragmentation():
    from llmkit.simulator.paged import ContiguousAllocator
    ca = ContiguousAllocator(capacity_tokens=8 * 4096, reserve_len=4096)
    ids = [f"s{i}" for i in range(8)]
    for i in ids:
        assert ca.allocate(i) is not None
    for i in ids[::2]:
        ca.free(i)
    st = ca.stats({i: 100 for i in ids[1::2]})
    assert st["free_tokens"] == 4 * 4096
    assert st["largest_free_run"] == 4096
    assert st["external_waste_pct"] == pytest.approx(75.0, abs=1.0)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_decode_is_memory_bound_and_prefill_is_compute_bound():
    from llmkit.simulator.cost import CostModel
    cm = CostModel(get_model("llama-3.1-8b"), get_gpu("h100-sxm"))
    assert cm.decode_ms([2048] * 32).bound_by == "memory"
    assert cm.prefill_ms([4096]).bound_by == "compute"


def test_batching_decode_is_nearly_free():
    """Weight traffic is constant in batch size: that is why batching works."""
    from llmkit.simulator.cost import CostModel
    cm = CostModel(get_model("llama-3.1-8b"), get_gpu("h100-sxm"))
    one = cm.decode_ms([2048]).duration_ms
    sixtyfour = cm.decode_ms([2048] * 64).duration_ms
    assert sixtyfour < one * 4           # 64x the work, <4x the time
    assert 64 / sixtyfour > 20 / one     # throughput improved enormously


def test_kv_reads_dominate_at_long_context():
    from llmkit.simulator.cost import CostModel
    cm = CostModel(get_model("llama-3.1-8b"), get_gpu("h100-sxm"))
    assert cm.roofline_summary(64, 512)["kv_share_of_traffic"] < 0.3
    assert cm.roofline_summary(64, 131072)["kv_share_of_traffic"] > 0.95


def test_prefill_is_superlinear_from_attention():
    from llmkit.simulator.cost import CostModel
    cm = CostModel(get_model("llama-3.1-8b"), get_gpu("h100-sxm"))
    t1 = cm.prefill_ms([4096]).duration_ms
    t2 = cm.prefill_ms([32768]).duration_ms
    assert t2 > t1 * 8                   # 8x tokens, more than 8x time


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------
def test_expected_tokens_is_truncated_geometric_not_k_alpha():
    from llmkit.specdec import expected_tokens
    alpha, k = 0.7, 8
    correct = expected_tokens(alpha, k)
    naive = k * alpha + 1
    assert correct == pytest.approx(3.199, abs=0.01)
    assert naive > correct * 2           # the naive count roughly doubles it


def test_speculation_has_an_optimal_k():
    from llmkit.specdec import optimal_k
    k, s = optimal_k(0.7, 0.16)
    assert 0 < k < 16
    assert s > 1.0


def test_speculation_loses_below_breakeven():
    from llmkit.specdec import breakeven_alpha, speedup
    be = breakeven_alpha(4, 0.16)
    assert speedup(be - 0.1, 4, 0.16) < 1.0
    assert speedup(be + 0.1, 4, 0.16) > 1.0


def test_speculation_gain_collapses_at_high_batch():
    from llmkit.specdec import plan
    low = max(p.speedup for p in plan(batch_size=1))
    high = max(p.speedup for p in plan(batch_size=64))
    assert low > high
    assert high < low * 0.8


# ---------------------------------------------------------------------------
# Engine simulator
# ---------------------------------------------------------------------------
def test_engine_completes_all_requests():
    from llmkit.simulator import EngineConfig, EngineSim, SimRequest
    e = EngineSim(EngineConfig(model="llama-3.1-8b", gpu="h100-sxm",
                               max_model_len=4096))
    e.submit([SimRequest(f"r{i}", prompt_tokens=512, output_tokens=32,
                         arrival_ms=i * 10.0) for i in range(16)])
    e.run()
    assert len(e.finished) == 16
    assert all(s.generated == 32 for s in e.finished)


def test_engine_records_are_summarisable():
    """The simulator and the real client must produce the same type."""
    from llmkit.simulator import EngineConfig, EngineSim, SimRequest
    e = EngineSim(EngineConfig(model="llama-3.1-8b", gpu="h100-sxm",
                               max_model_len=4096))
    e.submit([SimRequest(f"r{i}", prompt_tokens=512, output_tokens=32)
              for i in range(8)])
    e.run()
    s = summarize(e.records(), label="sim", concurrency=8, slo=SLO())
    assert s.n_ok == 8
    assert not math.isnan(s.ttft.p50)
    assert s.output_tok_per_s > 0


def test_shared_prefix_reduces_ttft():
    from llmkit.simulator import EngineConfig, EngineSim, SimRequest
    def run(shared):
        e = EngineSim(EngineConfig(model="llama-3.1-8b", gpu="h100-sxm",
                                   max_model_len=4096))
        e.submit([SimRequest(f"r{i}", prompt_tokens=1024, output_tokens=32,
                             arrival_ms=i * 20.0,
                             shared_prefix_tokens=shared,
                             prefix_group="g" if shared else None)
                  for i in range(32)])
        e.run()
        return summarize(e.records(), label="x", slo=SLO())
    cold = run(0)
    warm = run(512)
    assert warm.ttft.p50 < cold.ttft.p50
    assert warm.prefix_hit_rate > 0.4


def test_chunked_prefill_reduces_worst_decode_stall():
    """The core claim of project 08."""
    from llmkit.simulator import EngineConfig, EngineSim, SimRequest

    def run(chunked):
        e = EngineSim(EngineConfig(
            model="llama-3.1-8b", gpu="h100-sxm", max_model_len=20000,
            enable_chunked_prefill=chunked, max_num_batched_tokens=2048,
            enable_prefix_caching=False))
        reqs = [SimRequest(f"chat{i}", prompt_tokens=256, output_tokens=128,
                           arrival_ms=i * 5.0) for i in range(12)]
        reqs += [SimRequest(f"doc{j}", prompt_tokens=16384, output_tokens=16,
                            arrival_ms=300.0 + j * 400.0) for j in range(3)]
        e.submit(reqs)
        e.run()
        worst = 0.0
        for r in e.records():
            if r.request_id.startswith("chat"):
                itls = r.itls_ms
                if itls:
                    worst = max(worst, max(itls))
        return worst

    assert run(True) < run(False)


def test_preemption_occurs_under_memory_pressure():
    from llmkit.simulator import EngineConfig, EngineSim, SimRequest
    e = EngineSim(EngineConfig(
        model="llama-3.1-8b", gpu="h100-sxm", max_model_len=2048,
        num_gpu_blocks_override=200, max_num_seqs=64,
        enable_prefix_caching=False))
    e.submit([SimRequest(f"r{i}", prompt_tokens=1024, output_tokens=256,
                         arrival_ms=i * 1.0) for i in range(32)])
    e.run(max_steps=200_000)
    assert e.total_preemptions > 0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_prefix_affinity_is_stable_for_a_key():
    from llmkit.routing import PrefixAffinityRouter, Replica, prefix_key
    reps = [Replica(f"r{i}", f"http://h{i}") for i in range(4)]
    router = PrefixAffinityRouter(reps)
    k = prefix_key("system", "user")
    assert len({router.pick(reps, k).name for _ in range(20)}) == 1


def test_prefix_affinity_deflects_when_overloaded():
    from llmkit.routing import PrefixAffinityRouter, Replica, prefix_key
    reps = [Replica(f"r{i}", f"http://h{i}") for i in range(4)]
    router = PrefixAffinityRouter(reps, overload_factor=1.1)
    k = prefix_key("system", "user")
    preferred = router.pick(reps, k)
    preferred.inflight = 500
    assert router.pick(reps, k).name != preferred.name


def test_consistent_hash_moves_about_one_nth_of_keys():
    from llmkit.routing import ConsistentHashRing, Replica, prefix_key
    reps = [Replica(f"r{i}", f"http://h{i}") for i in range(8)]
    ring = ConsistentHashRing(reps)
    keys = [prefix_key(f"s{i}", "u") for i in range(2000)]
    before = {k: ring.lookup(k)[0] for k in keys}
    ring.rebuild(reps[:7])
    after = {k: ring.lookup(k)[0] for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k]) / len(keys)
    assert 0.08 < moved < 0.22        # ideal 1/8 = 0.125


def test_breaker_opens_after_consecutive_errors():
    from llmkit.routing import Replica
    r = Replica("r0", "http://h0")
    for _ in range(5):
        r.observe_error(breaker_threshold=5, open_s=30)
    assert not r.available
    r.observe_success()
    assert r.consecutive_errors == 0


# ---------------------------------------------------------------------------
# Gateway primitives
# ---------------------------------------------------------------------------
def test_token_bucket_limits_and_refills():
    from llmkit.gateway import TokenBucket
    b = TokenBucket(rate=10, capacity=10)
    assert all(b.try_consume(1) for _ in range(10))
    assert not b.try_consume(1)
    assert b.retry_after_s(1) > 0


def test_tenant_limits_cover_both_dimensions():
    from llmkit.gateway import TenantLimits
    t = TenantLimits("t", rpm=6000, tpm=600, burst_requests=100, burst_tokens=100)
    ok, reason, _ = t.check(1000)          # far over the token burst
    assert not ok and reason == "tpm"
    ok, reason, _ = t.check(10)
    assert ok, "a token rejection must not consume the request allowance"


def test_retry_budget_caps_amplification():
    from llmkit.gateway import RetryBudget
    b = RetryBudget(ratio=0.1, window_s=60, min_per_s=0.0)
    for _ in range(100):
        b.record_request()
    allowed = sum(1 for _ in range(100) if b.try_retry())
    assert allowed <= 11
    assert b.denied > 0


def test_degradation_chain_orders_tiers():
    from llmkit.gateway import Backend, DegradationChain
    chain = DegradationChain(backends=[
        Backend("cheap", "u", "m", tier="cheaper_model"),
        Backend("main", "u", "m", tier="primary"),
        Backend("api", "u", "m", tier="secondary"),
    ])
    names = [b.name for b in chain.candidates()]
    assert names == ["main", "api", "cheap"]


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def test_decode_mfu_is_structurally_low_but_mbu_is_not():
    from llmkit.costs import compute_utilization
    u = compute_utilization("llama-3.1-8b", "h100-sxm", window_s=60,
                            prompt_tokens=100_000, output_tokens=200_000,
                            mean_context=2048, mean_batch=64)
    assert u.decode_mfu < 0.10
    assert u.decode_mbu > u.decode_mfu * 5


def test_cached_prompt_tokens_excluded_from_prefill_flops():
    from llmkit.costs import compute_utilization
    a = compute_utilization("llama-3.1-8b", "h100-sxm", window_s=60,
                            prompt_tokens=100_000, output_tokens=1000,
                            cached_prompt_tokens=0)
    b = compute_utilization("llama-3.1-8b", "h100-sxm", window_s=60,
                            prompt_tokens=100_000, output_tokens=1000,
                            cached_prompt_tokens=90_000)
    assert b.prefill_flops < a.prefill_flops / 5


def test_breakeven_reports_required_utilization():
    from llmkit.costs import breakeven_vs_api
    r = breakeven_vs_api(0.30, 0.60)
    assert r["min_utilization_to_break_even"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Chaos / SLO
# ---------------------------------------------------------------------------
def test_burn_rate_matches_sre_thresholds():
    from llmkit.chaos import SLOTarget
    slo = SLOTarget(target=0.99)
    assert slo.burn_rate(0.01) == pytest.approx(1.0)
    assert slo.burn_rate(0.144) == pytest.approx(14.4, abs=0.01)
    assert "page immediately" in slo.severity(14.4)
    assert slo.severity(0.5) == "within budget"


# ---------------------------------------------------------------------------
# Autoscaling
# ---------------------------------------------------------------------------
def test_queue_depth_beats_gpu_util_on_a_spike():
    from llmkit.autoscale import ScalerConfig, simulate, spike
    traffic = spike(base=5, peak=60, at_s=600, width_s=200)
    q = simulate(ScalerConfig(signal="queue_depth", target=8.0, max_replicas=20),
                 traffic, duration_s=1800, capacity_per_replica_rps=10)
    g = simulate(ScalerConfig(signal="gpu_util", target=70.0, max_replicas=20),
                 traffic, duration_s=1800, capacity_per_replica_rps=10)
    assert q.summary()["slo_violation_s"] <= g.summary()["slo_violation_s"]


def test_faster_cold_start_reduces_slo_burn():
    from llmkit.autoscale import ScalerConfig, simulate, spike
    traffic = spike(base=5, peak=60, at_s=600, width_s=200)
    slow = simulate(ScalerConfig(weight_load_s=200.0), traffic,
                    duration_s=1800, capacity_per_replica_rps=10)
    fast = simulate(ScalerConfig(image_pull_s=0.0, weight_load_s=20.0), traffic,
                    duration_s=1800, capacity_per_replica_rps=10)
    assert fast.summary()["slo_violation_s"] <= slow.summary()["slo_violation_s"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_publication_gate_rejects_incomplete_metadata():
    from llmkit import RunMeta
    meta = RunMeta(run_id="x", model="", engine="unknown")
    problems = meta.validate()
    assert any("model" in p for p in problems)
    assert any("engine" in p for p in problems)


def test_publication_gate_flags_heuristic_tokens():
    from llmkit import RunMeta
    meta = RunMeta(run_id="x", model="m", engine="vllm",
                   token_source="heuristic", simulated=True)
    assert any("estimated" in p for p in meta.validate())
