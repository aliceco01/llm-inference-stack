"""Continuous-batching engine simulator on a virtual clock.

What this models, because these are the behaviours that produce the latency
curves people actually debug:

* Continuous batching: sequences join and leave the running batch at step
  granularity, not request granularity.
* Prefill-priority vs chunked prefill scheduling, as two selectable policies,
  because the difference between them IS project 08.
* Paged KV allocation with prefix reuse, so a warm prefix skips both the blocks
  and the prefill compute.
* Preemption under memory pressure, by recompute or by swap, with the cost of
  each actually charged to the clock.
* Speculative decoding with a configurable acceptance rate.

What it does not model: numerics, actual attention output, tokenizer effects,
NUMA, or kernel autotuning. It predicts *timing and memory behaviour*, and its
outputs are always tagged simulated=True.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..gpus import GPUSpec, get_gpu
from ..kvcache import ServingConfig, compute_budget
from ..modelspec import ModelSpec, get_model
from ..types import FinishReason, RequestRecord, TokenEvent
from .cost import CostModel, EfficiencyModel
from .paged import OutOfBlocks, PagedKVCache

MS = 1_000_000  # ns per ms


def synth_token_ids(r: SimRequest) -> list[int]:
    """Deterministic token ids with a controllable shared prefix.

    Distinct requests must produce distinct token ids, or every request
    trivially hits the prefix cache and the simulator reports a speedup that
    would never occur in production. Sharing is therefore opt-in: only the
    first `shared_prefix_tokens` are derived from the group/session key, and
    everything after them is derived from the request id.
    """
    shared_key = r.prefix_group or r.session_id
    n_shared = min(r.shared_prefix_tokens, r.prompt_tokens) if shared_key else 0
    ids: list[int] = []
    if n_shared:
        base = (abs(hash(("prefix", shared_key))) % 1_000_000) * 1000
        ids.extend(base + i for i in range(n_shared))
    # Multi-turn sessions additionally share the growing conversation history,
    # so turn k of a session repeats turn k-1's tokens exactly.
    if r.session_id and not r.prefix_group:
        base = (abs(hash(("sess", r.session_id))) % 1_000_000) * 1000 + 500_000_000
        ids.extend(base + i for i in range(n_shared, r.prompt_tokens))
        return ids[: r.prompt_tokens]
    base = (abs(hash(("uniq", r.request_id))) % 1_000_000) * 1000 + 900_000_000
    ids.extend(base + i for i in range(r.prompt_tokens - len(ids)))
    return ids[: r.prompt_tokens]


class SeqStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


class PreemptionMode(str, Enum):
    RECOMPUTE = "recompute"
    SWAP = "swap"


@dataclass
class SimRequest:
    """Input to the simulator."""

    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_ms: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    session_id: str | None = None
    tenant: str | None = None
    priority: int = 0            # lower runs first when priority scheduling is on
    # Tokens at the head of the prompt that are byte-identical to every other
    # request in `prefix_group`. This is the knob prefix-cache experiments turn.
    shared_prefix_tokens: int = 0
    prefix_group: str | None = None


@dataclass
class Sequence:
    """Engine-side state for one in-flight request."""

    req: SimRequest
    status: SeqStatus = SeqStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    prefilled: int = 0            # prompt tokens whose KV is computed
    generated: int = 0
    cached_tokens: int = 0        # prompt tokens served from prefix cache
    arrival_ms: float = 0.0
    first_token_ms: float = -1.0
    finish_ms: float = -1.0
    token_times_ms: list[float] = field(default_factory=list)
    n_preemptions: int = 0
    n_spec_accepted: int = 0
    n_spec_proposed: int = 0

    @property
    def ctx_len(self) -> int:
        return self.prefilled + self.generated

    @property
    def total_len(self) -> int:
        return self.req.prompt_tokens + self.generated

    @property
    def prefill_done(self) -> bool:
        return self.prefilled >= self.req.prompt_tokens

    @property
    def done(self) -> bool:
        return self.generated >= self.req.output_tokens


@dataclass
class EngineConfig:
    model: str = "llama-3.1-8b"
    gpu: str = "h100-sxm"
    tp: int = 1
    pp: int = 1
    weight_dtype: str = "bf16"
    kv_dtype: str = "fp16"
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    max_model_len: int = 8192
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    enable_chunked_prefill: bool = False
    long_prefill_token_threshold: int = 0   # 0 = no per-request chunk cap
    enable_prefix_caching: bool = True
    preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE
    scheduling: str = "fcfs"                # fcfs | priority
    cuda_graphs: bool = True

    # speculative decoding
    spec_draft_tokens: int = 0
    spec_acceptance_rate: float = 0.0
    spec_draft_cost_ratio: float = 0.06     # draft model step cost vs target

    # optional hard override of block count (for memory-pressure experiments)
    num_gpu_blocks_override: int | None = None
    swap_bandwidth_gb_s: float = 25.0       # PCIe gen4 x16 practical
    seed: int = 0


@dataclass
class StepTrace:
    """One scheduler step, recorded for the timeline plots in projects 08/09."""

    step: int
    t_ms: float
    dt_ms: float
    phase: str
    prefill_tokens: int = 0
    decode_seqs: int = 0
    n_running: int = 0
    n_waiting: int = 0
    n_swapped: int = 0
    kv_util: float = 0.0
    preemptions: int = 0
    cache_hit_blocks: int = 0
    bound_by: str = ""


class EngineSim:
    """Single-replica engine. Deterministic given a seed."""

    def __init__(self, cfg: EngineConfig, *, efficiency: EfficiencyModel | None = None) -> None:
        self.cfg = cfg
        self.model: ModelSpec = get_model(cfg.model)
        self.gpu: GPUSpec = get_gpu(cfg.gpu)
        self.rng = random.Random(cfg.seed)

        sc = ServingConfig(
            tp=cfg.tp, pp=cfg.pp, weight_dtype=cfg.weight_dtype, kv_dtype=cfg.kv_dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization, block_size=cfg.block_size,
            max_model_len=cfg.max_model_len, max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_cuda_graphs=cfg.cuda_graphs, enable_prefix_caching=cfg.enable_prefix_caching,
        )
        self.budget = compute_budget(self.model, self.gpu, sc)
        n_blocks = cfg.num_gpu_blocks_override or self.budget.num_blocks
        if n_blocks <= 0:
            raise ValueError(
                f"configuration leaves no KV cache:\n{self.budget.explain()}"
            )
        self.kv = PagedKVCache(
            n_blocks, cfg.block_size,
            enable_prefix_caching=cfg.enable_prefix_caching,
        )
        self.cost = CostModel(
            self.model, self.gpu, tp=cfg.tp, pp=cfg.pp,
            weight_dtype=cfg.weight_dtype, kv_dtype=cfg.kv_dtype,
            eff=efficiency, cuda_graphs=cfg.cuda_graphs,
        )

        self.now_ms = 0.0
        self.step_no = 0
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.swapped: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.traces: list[StepTrace] = []
        self.total_preemptions = 0
        self._pending: list[tuple[float, int, SimRequest]] = []
        self._seq_counter = 0

    # ------------------------------------------------------------------
    def submit(self, reqs: Iterable[SimRequest]) -> None:
        for r in reqs:
            self._seq_counter += 1
            heapq.heappush(self._pending, (r.arrival_ms, self._seq_counter, r))

    def _admit_arrivals(self) -> None:
        while self._pending and self._pending[0][0] <= self.now_ms + 1e-9:
            _, _, r = heapq.heappop(self._pending)
            s = Sequence(req=r, arrival_ms=r.arrival_ms)
            if not r.token_ids:
                r.token_ids = synth_token_ids(r)
            self.waiting.append(s)

    # ------------------------------------------------------------------
    def _allocate(self, s: Sequence) -> bool:
        """Give a waiting sequence its prompt blocks. False if memory is short."""
        need = math.ceil(s.req.prompt_tokens / self.cfg.block_size)
        if not self.kv.can_allocate(need):
            return False
        try:
            table, cached = self.kv.allocate_prompt(s.req.token_ids)
        except OutOfBlocks:
            return False
        s.block_table = table
        s.cached_tokens = cached
        # Cached prompt tokens skip prefill compute entirely: that is where the
        # TTFT saving in project 04 comes from.
        s.prefilled = cached
        return True

    def _preempt(self, victim: Sequence) -> float:
        """Evict a running sequence. Returns the time cost in ms."""
        self.total_preemptions += 1
        victim.n_preemptions += 1
        cost = 0.0
        if self.cfg.preemption_mode == PreemptionMode.SWAP:
            kv_bytes = victim.ctx_len * self.model.kv_bytes_per_token(self.cfg.kv_dtype)
            cost = kv_bytes / (self.cfg.swap_bandwidth_gb_s * 1e9) * 1e3
            self.kv.free(victim.block_table)
            victim.block_table = []
            victim.status = SeqStatus.SWAPPED
            self.swapped.append(victim)
        else:
            # Recompute: throw the KV away and re-prefill from scratch later.
            # Cheap to evict, expensive to restart, and the restart cost is
            # paid in TTFT of an already-running request.
            self.kv.free(victim.block_table)
            victim.block_table = []
            victim.prefilled = 0
            victim.generated = 0
            victim.first_token_ms = -1.0
            victim.token_times_ms.clear()
            victim.status = SeqStatus.WAITING
            self.waiting.insert(0, victim)
        self.running.remove(victim)
        return cost

    def _sort_waiting(self) -> None:
        if self.cfg.scheduling == "priority":
            self.waiting.sort(key=lambda s: (s.req.priority, s.arrival_ms))

    # ------------------------------------------------------------------
    def _schedule(self) -> tuple[list[tuple[Sequence, int]], list[Sequence], float]:
        """Choose this step's work.

        Returns (prefill_chunks, decode_seqs, extra_ms_charged).
        """
        cfg = self.cfg
        self._sort_waiting()
        extra_ms = 0.0
        token_budget = cfg.max_num_batched_tokens
        prefills: list[tuple[Sequence, int]] = []

        if cfg.enable_chunked_prefill:
            # Decodes are admitted first and each costs one token of budget, so
            # decode never starves. Prefill chunks then consume what is left.
            decodes = [s for s in self.running if s.prefill_done]
            token_budget -= len(decodes)
            partial = [s for s in self.running if not s.prefill_done]
            for s in partial:
                if token_budget <= 0:
                    break
                remaining = s.req.prompt_tokens - s.prefilled
                cap = cfg.long_prefill_token_threshold or token_budget
                chunk = min(remaining, token_budget, cap)
                if chunk > 0:
                    prefills.append((s, chunk))
                    token_budget -= chunk
            while (self.waiting and token_budget > 0
                   and len(self.running) + len(prefills) < cfg.max_num_seqs):
                s = self.waiting[0]
                if not self._allocate(s):
                    break
                self.waiting.pop(0)
                s.status = SeqStatus.RUNNING
                self.running.append(s)
                remaining = s.req.prompt_tokens - s.prefilled
                if remaining <= 0:
                    continue   # fully prefix-cached: straight to decode
                cap = cfg.long_prefill_token_threshold or token_budget
                chunk = min(remaining, token_budget, cap)
                prefills.append((s, chunk))
                token_budget -= chunk
            return prefills, decodes, extra_ms

        # --- prefill-priority (vLLM default without chunked prefill) -------
        # If anything is waiting and fits, the whole step is a prefill step and
        # every running sequence is stalled for its duration. This is the
        # decode starvation that project 08 measures.
        admitted = False
        while (self.waiting and len(self.running) < cfg.max_num_seqs
               and token_budget > 0):
            s = self.waiting[0]
            need = s.req.prompt_tokens - s.cached_tokens
            if need > token_budget and admitted:
                break
            if not self._allocate(s):
                break
            self.waiting.pop(0)
            s.status = SeqStatus.RUNNING
            self.running.append(s)
            chunk = s.req.prompt_tokens - s.prefilled
            if chunk > 0:
                prefills.append((s, chunk))
                token_budget -= chunk
            admitted = True
            if token_budget <= 0:
                break
        if prefills:
            return prefills, [], extra_ms

        # Restore swapped sequences when there is room.
        while self.swapped and len(self.running) < cfg.max_num_seqs:
            s = self.swapped[0]
            need = math.ceil(s.total_len / cfg.block_size)
            if not self.kv.can_allocate(need):
                break
            self.swapped.pop(0)
            table, cached = self.kv.allocate_prompt(
                s.req.token_ids[: s.total_len] or [0] * s.total_len
            )
            s.block_table = table
            s.status = SeqStatus.RUNNING
            kv_bytes = s.ctx_len * self.model.kv_bytes_per_token(cfg.kv_dtype)
            extra_ms += kv_bytes / (cfg.swap_bandwidth_gb_s * 1e9) * 1e3
            self.running.append(s)

        decodes = [s for s in self.running if s.prefill_done]
        return [], decodes, extra_ms

    # ------------------------------------------------------------------
    def _grow_or_preempt(self, decodes: list[Sequence]) -> tuple[list[Sequence], float]:
        """Append one token of KV per decoding sequence, preempting if needed.

        Preemption victim is the newest sequence (LIFO), matching vLLM: the
        oldest requests are closest to finishing, so evicting them wastes the
        most work and inflates tail latency the worst.
        """
        extra_ms = 0.0
        survivors: list[Sequence] = []
        for s in decodes:
            placed = False
            while not placed:
                try:
                    self.kv.append_token(s.block_table, s.total_len)
                    placed = True
                except OutOfBlocks:
                    victim = None
                    for cand in reversed(self.running):
                        if cand is not s:
                            victim = cand
                            break
                    if victim is None:
                        extra_ms += self._preempt(s)
                        break
                    extra_ms += self._preempt(victim)
                    if victim in survivors:
                        survivors.remove(victim)
            if placed:
                survivors.append(s)
        return survivors, extra_ms

    # ------------------------------------------------------------------
    def step(self) -> StepTrace | None:
        self._admit_arrivals()
        if not (self.running or self.waiting or self.swapped):
            if not self._pending:
                return None
            # Idle: jump the clock to the next arrival rather than spinning.
            self.now_ms = self._pending[0][0]
            self._admit_arrivals()

        prefills, decodes, extra_ms = self._schedule()
        hits_before = self.kv.stats.cache_hits
        preempt_before = self.total_preemptions

        if decodes:
            decodes, grow_ms = self._grow_or_preempt(decodes)
            extra_ms += grow_ms

        if not prefills and not decodes:
            # Nothing runnable: memory is fully committed and every candidate
            # was preempted. Advance a tick so the loop cannot spin forever.
            self.now_ms += 0.1
            self.step_no += 1
            return StepTrace(self.step_no, self.now_ms, 0.1, "stalled",
                             n_running=len(self.running), n_waiting=len(self.waiting),
                             n_swapped=len(self.swapped), kv_util=self.kv.utilization)

        chunk_lens = [c for _, c in prefills]
        prefill_ctx = [s.prefilled + c for s, c in prefills]
        decode_ctx = [s.ctx_len for s in decodes]

        if prefills and decodes:
            sc = self.cost.mixed_ms(chunk_lens, prefill_ctx, decode_ctx)
            phase = "mixed"
        elif prefills:
            sc = self.cost.prefill_ms(chunk_lens, prefill_ctx)
            phase = "prefill"
        else:
            sc = self.cost.decode_ms(decode_ctx)
            phase = "decode"

        dt = sc.duration_ms + extra_ms
        tokens_this_step = 1

        # --- speculative decoding ------------------------------------------
        if self.cfg.spec_draft_tokens > 0 and decodes and not prefills:
            k = self.cfg.spec_draft_tokens
            draft_ms = sc.duration_ms * self.cfg.spec_draft_cost_ratio * k
            # One target forward verifies all k drafts plus the bonus token.
            verify = self.cost.decode_ms([c + k for c in decode_ctx])
            dt = draft_ms + verify.duration_ms + extra_ms
            accepted = self._sample_accepted(k)
            tokens_this_step = accepted + 1
            for s in decodes:
                s.n_spec_proposed += k
                s.n_spec_accepted += accepted

        self.now_ms += dt
        self.step_no += 1

        # --- apply results --------------------------------------------------
        for s, c in prefills:
            s.prefilled += c
        for s in decodes:
            emitted = min(tokens_this_step, s.req.output_tokens - s.generated)
            for i in range(emitted):
                # Tokens inside one speculative batch land together; spacing
                # them evenly would understate the burstiness that users see.
                s.token_times_ms.append(self.now_ms)
            s.generated += emitted
            if s.first_token_ms < 0 and s.generated > 0:
                s.first_token_ms = self.now_ms
        # A sequence that just finished prefill emits its first token this step.
        for s, c in prefills:
            if s.prefill_done and s.first_token_ms < 0:
                s.generated = 1
                s.first_token_ms = self.now_ms
                s.token_times_ms.append(self.now_ms)

        for s in list(self.running):
            if s.done:
                s.finish_ms = self.now_ms
                s.status = SeqStatus.FINISHED
                self.kv.free(s.block_table)
                s.block_table = []
                self.running.remove(s)
                self.finished.append(s)

        self.kv.step = self.step_no
        tr = StepTrace(
            step=self.step_no, t_ms=self.now_ms, dt_ms=dt, phase=phase,
            prefill_tokens=sum(chunk_lens), decode_seqs=len(decodes),
            n_running=len(self.running), n_waiting=len(self.waiting),
            n_swapped=len(self.swapped), kv_util=self.kv.utilization,
            preemptions=self.total_preemptions - preempt_before,
            cache_hit_blocks=self.kv.stats.cache_hits - hits_before,
            bound_by=sc.bound_by,
        )
        self.traces.append(tr)
        return tr

    def run(self, max_steps: int = 2_000_000, max_ms: float = 1e9) -> None:
        while self.step_no < max_steps and self.now_ms < max_ms:
            if self.step() is None:
                break

    def _sample_accepted(self, k: int) -> int:
        """Accepted draft tokens this step.

        Rejection sampling stops at the FIRST rejection, so with per-token
        acceptance p the expected accepted count is sum_{i=1..k} p^i, not k*p.
        Modelling it as k*p overstates the speedup substantially at large k,
        which is the usual way speculative decoding gets oversold.
        """
        p = self.cfg.spec_acceptance_rate
        n = 0
        for _ in range(k):
            if self.rng.random() < p:
                n += 1
            else:
                break
        return n

    # ------------------------------------------------------------------
    def records(self, *, t0_ns: int = 0) -> list[RequestRecord]:
        """Convert finished sequences into the repo's canonical record type."""
        out: list[RequestRecord] = []
        for s in self.finished + self.running:
            r = RequestRecord(
                request_id=s.req.request_id,
                model=self.cfg.model,
                session_id=s.req.session_id,
                tenant=s.req.tenant,
                replica=f"sim-{self.cfg.gpu}",
            )
            r.t_submit_ns = t0_ns + int(s.arrival_ms * MS)
            r.t_send_ns = r.t_submit_ns
            r.prompt_tokens = s.req.prompt_tokens
            r.output_tokens = s.generated
            r.cached_prompt_tokens = s.cached_tokens
            if s.token_times_ms:
                r.t_first_chunk_ns = t0_ns + int(s.token_times_ms[0] * MS)
                r.t_first_token_ns = r.t_first_chunk_ns
                r.t_last_token_ns = t0_ns + int(s.token_times_ms[-1] * MS)
                r.token_events = [
                    TokenEvent(t_ns=t0_ns + int(t * MS)) for t in s.token_times_ms
                ]
            r.t_done_ns = t0_ns + int(
                (s.finish_ms if s.finish_ms > 0 else self.now_ms) * MS
            )
            r.finish_reason = (
                FinishReason.STOP if s.done else FinishReason.CANCELLED
            )
            r.extra.update({
                "simulated": True,
                "preemptions": s.n_preemptions,
                "spec_accepted": s.n_spec_accepted,
                "spec_proposed": s.n_spec_proposed,
            })
            out.append(r)
        return out

    def stats(self) -> dict[str, Any]:
        acc = sum(s.n_spec_accepted for s in self.finished)
        prop = sum(s.n_spec_proposed for s in self.finished)
        phases: dict[str, float] = {}
        for t in self.traces:
            phases[t.phase] = phases.get(t.phase, 0.0) + t.dt_ms
        return {
            "steps": self.step_no,
            "sim_time_ms": round(self.now_ms, 2),
            "finished": len(self.finished),
            "running": len(self.running),
            "waiting": len(self.waiting),
            "swapped": len(self.swapped),
            "preemptions": self.total_preemptions,
            "kv": self.kv.snapshot(),
            "num_gpu_blocks": self.kv.num_blocks,
            "kv_cache_gib": round(self.budget.kv_cache_gib, 2),
            "time_by_phase_ms": {k: round(v, 1) for k, v in phases.items()},
            "spec_acceptance": round(acc / prop, 3) if prop else None,
        }
