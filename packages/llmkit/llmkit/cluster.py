"""Cluster-level simulation: colocated vs disaggregated prefill/decode.

Why disaggregation exists
-------------------------
Prefill is compute bound and decode is memory-bandwidth bound. Running both on
the same GPU means they contend for different resources while sharing one
scheduler, and neither phase gets a machine tuned for it. Splitting them gives
three things:

1. **No interference.** Decode never stalls behind a prefill step, without
   needing chunked prefill to paper over it.
2. **Independent scaling.** Prefill and decode demand scale with different
   things (input tokens vs output tokens x concurrency), so a fixed ratio
   inside one replica is nearly always the wrong ratio.
3. **Heterogeneous hardware.** Prefill wants FLOPs, decode wants HBM bandwidth
   and capacity. You can buy each separately.

What it costs
-------------
The KV cache computed during prefill has to reach the decode worker. That
transfer is the entire tradeoff, and it is not small: an 8B model at 8k context
produces 1 GiB of KV. Over a 400 Gb/s RDMA link at realistic efficiency that is
roughly 25 ms added to TTFT, every request.

So disaggregation pays when prefill/decode interference costs more than the
transfer, which is workload dependent: long prompts with long generations win,
short prompts with short generations lose. This module computes where the line
is rather than assuming it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .gpus import GPUSpec, get_gpu
from .modelspec import ModelSpec, get_model
from .simulator.cost import CostModel, EfficiencyModel

Topology = Literal["colocated", "disaggregated"]


@dataclass
class Interconnect:
    """Link between prefill and decode pools."""

    name: str = "rdma-400g"
    gb_s: float = 50.0            # achieved, not line rate
    latency_us: float = 15.0      # one-way setup latency per transfer

    def transfer_ms(self, nbytes: float) -> float:
        return self.latency_us / 1000.0 + nbytes / (self.gb_s * 1e9) * 1e3


INTERCONNECTS: dict[str, Interconnect] = {
    "nvlink": Interconnect("nvlink", gb_s=400.0, latency_us=2.0),
    "rdma-400g": Interconnect("rdma-400g", gb_s=50.0, latency_us=15.0),
    "rdma-200g": Interconnect("rdma-200g", gb_s=25.0, latency_us=15.0),
    "rdma-100g": Interconnect("rdma-100g", gb_s=12.0, latency_us=20.0),
    "tcp-25g": Interconnect("tcp-25g", gb_s=2.5, latency_us=120.0),
    # Shared storage / object tiering, as used for cross-node prefix reuse.
    "nvme-tier": Interconnect("nvme-tier", gb_s=6.0, latency_us=300.0),
}


@dataclass
class ClusterRequest:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_ms: float = 0.0

    # filled in by the simulation
    prefill_start_ms: float = -1.0
    prefill_done_ms: float = -1.0
    transfer_done_ms: float = -1.0
    first_token_ms: float = -1.0
    finish_ms: float = -1.0
    prefill_worker: int = -1
    decode_worker: int = -1

    @property
    def ttft_ms(self) -> float:
        return self.first_token_ms - self.arrival_ms if self.first_token_ms >= 0 else math.nan

    @property
    def e2e_ms(self) -> float:
        return self.finish_ms - self.arrival_ms if self.finish_ms >= 0 else math.nan


@dataclass
class ClusterConfig:
    model: str = "llama-3.1-8b"
    prefill_gpu: str = "h100-sxm"
    decode_gpu: str = "h100-sxm"
    n_prefill: int = 1
    n_decode: int = 1
    prefill_tp: int = 1
    decode_tp: int = 1
    interconnect: str = "rdma-400g"
    kv_dtype: str = "fp16"
    # colocated only: how the single pool splits its steps
    max_num_batched_tokens: int = 8192
    max_batch: int = 256
    chunked_prefill: bool = True


@dataclass
class ClusterResult:
    topology: str
    n_prefill: int
    n_decode: int
    requests: list[ClusterRequest] = field(default_factory=list)
    makespan_ms: float = 0.0
    prefill_busy_ms: list[float] = field(default_factory=list)
    decode_busy_ms: list[float] = field(default_factory=list)
    transfer_ms_total: float = 0.0
    notes: list[str] = field(default_factory=list)

    def ttfts(self) -> list[float]:
        return [r.ttft_ms for r in self.requests if not math.isnan(r.ttft_ms)]

    def e2es(self) -> list[float]:
        return [r.e2e_ms for r in self.requests if not math.isnan(r.e2e_ms)]

    def output_tok_per_s(self) -> float:
        toks = sum(r.output_tokens for r in self.requests if r.finish_ms >= 0)
        return toks / max(self.makespan_ms / 1000.0, 1e-9)

    def utilization(self) -> dict[str, float]:
        span = max(self.makespan_ms, 1e-9)
        return {
            "prefill_pool": (sum(self.prefill_busy_ms) /
                             max(len(self.prefill_busy_ms), 1) / span),
            "decode_pool": (sum(self.decode_busy_ms) /
                            max(len(self.decode_busy_ms), 1) / span),
        }

    def summary(self) -> dict[str, Any]:
        from .metrics import percentile
        t, e = self.ttfts(), self.e2es()
        u = self.utilization()
        return {
            "topology": self.topology,
            "prefill_workers": self.n_prefill,
            "decode_workers": self.n_decode,
            "requests": len(self.requests),
            "makespan_ms": round(self.makespan_ms, 1),
            "ttft_p50": round(percentile(t, 50), 1) if t else math.nan,
            "ttft_p95": round(percentile(t, 95), 1) if t else math.nan,
            "e2e_p50": round(percentile(e, 50), 1) if e else math.nan,
            "e2e_p95": round(percentile(e, 95), 1) if e else math.nan,
            "out_tok_per_s": round(self.output_tok_per_s(), 1),
            "prefill_util": round(u["prefill_pool"], 3),
            "decode_util": round(u["decode_pool"], 3),
            "transfer_ms_total": round(self.transfer_ms_total, 1),
        }


class ClusterSim:
    """Coarse-grained cluster model.

    Deliberately coarser than `EngineSim`: it models queueing between pools and
    the KV transfer, treating each worker's internal behaviour with the same
    roofline cost model but without per-block accounting. The question it
    answers is "how should the fleet be shaped", not "what does the block
    allocator do", and mixing those two resolutions would make it slow without
    making it more accurate.
    """

    def __init__(self, cfg: ClusterConfig, *, efficiency: EfficiencyModel | None = None) -> None:
        self.cfg = cfg
        self.model: ModelSpec = get_model(cfg.model)
        self.pgpu: GPUSpec = get_gpu(cfg.prefill_gpu)
        self.dgpu: GPUSpec = get_gpu(cfg.decode_gpu)
        self.link = INTERCONNECTS.get(cfg.interconnect, INTERCONNECTS["rdma-400g"])
        self.pcost = CostModel(self.model, self.pgpu, tp=cfg.prefill_tp,
                               kv_dtype=cfg.kv_dtype, eff=efficiency)
        self.dcost = CostModel(self.model, self.dgpu, tp=cfg.decode_tp,
                               kv_dtype=cfg.kv_dtype, eff=efficiency)

    def kv_transfer_bytes(self, prompt_tokens: int) -> float:
        return prompt_tokens * self.model.kv_bytes_per_token(self.cfg.kv_dtype)

    # ------------------------------------------------------------------
    def run_disaggregated(self, reqs: list[ClusterRequest]) -> ClusterResult:
        """Prefill pool -> KV transfer -> decode pool.

        Prefill workers are modelled as serving one request at a time to
        completion (prefill is compute bound, so batching several long prompts
        does not help much and the scheduling is simpler to reason about).
        Decode workers run continuous batching over whatever they hold.
        """
        cfg = self.cfg
        res = ClusterResult("disaggregated", cfg.n_prefill, cfg.n_decode)
        res.prefill_busy_ms = [0.0] * cfg.n_prefill
        res.decode_busy_ms = [0.0] * cfg.n_decode

        # --- stage 1: prefill, FCFS across a pool of workers ---------------
        free_p: list[tuple[float, int]] = [(0.0, i) for i in range(cfg.n_prefill)]
        heapq.heapify(free_p)
        ready_for_decode: list[tuple[float, ClusterRequest]] = []
        for r in sorted(reqs, key=lambda x: x.arrival_ms):
            avail, w = heapq.heappop(free_p)
            start = max(avail, r.arrival_ms)
            dur = self.pcost.prefill_ms([r.prompt_tokens]).duration_ms
            r.prefill_worker = w
            r.prefill_start_ms = start
            r.prefill_done_ms = start + dur
            res.prefill_busy_ms[w] += dur
            heapq.heappush(free_p, (r.prefill_done_ms, w))

            xfer = self.link.transfer_ms(self.kv_transfer_bytes(r.prompt_tokens))
            r.transfer_done_ms = r.prefill_done_ms + xfer
            res.transfer_ms_total += xfer
            ready_for_decode.append((r.transfer_done_ms, r))

        # --- stage 2: decode pool, least-loaded assignment -----------------
        ready_for_decode.sort(key=lambda t: t[0])
        # Each decode worker runs a continuous batch. We step each worker's
        # local clock, admitting requests as they arrive.
        per_worker: list[list[ClusterRequest]] = [[] for _ in range(cfg.n_decode)]
        counts = [0] * cfg.n_decode
        for _, r in ready_for_decode:
            w = min(range(cfg.n_decode), key=lambda i: counts[i])
            r.decode_worker = w
            counts[w] += 1
            per_worker[w].append(r)

        for w, group in enumerate(per_worker):
            self._decode_worker(group, res, w)

        res.makespan_ms = max((r.finish_ms for r in reqs if r.finish_ms >= 0),
                              default=0.0)
        res.requests = reqs
        if cfg.n_prefill and cfg.n_decode:
            u = res.utilization()
            if u["prefill_pool"] < 0.5 and u["decode_pool"] > 0.9:
                res.notes.append(
                    f"prefill pool only {u['prefill_pool']*100:.0f}% utilised while "
                    f"decode is saturated: shift GPUs from prefill to decode")
            if u["decode_pool"] < 0.5 and u["prefill_pool"] > 0.9:
                res.notes.append(
                    f"decode pool only {u['decode_pool']*100:.0f}% utilised while "
                    f"prefill is saturated: shift GPUs from decode to prefill")
        return res

    def _decode_worker(self, group: list[ClusterRequest], res: ClusterResult,
                       w: int) -> None:
        """Continuous-batching decode on one worker, stepped on a virtual clock."""
        if not group:
            return
        pending = sorted(group, key=lambda r: r.transfer_done_ms)
        idx = 0
        active: list[ClusterRequest] = []
        remaining: dict[str, int] = {}
        ctx: dict[str, int] = {}
        now = pending[0].transfer_done_ms

        while idx < len(pending) or active:
            while (idx < len(pending)
                   and pending[idx].transfer_done_ms <= now
                   and len(active) < self.cfg.max_batch):
                r = pending[idx]; idx += 1
                active.append(r)
                remaining[r.request_id] = r.output_tokens
                ctx[r.request_id] = r.prompt_tokens
                # First token is emitted by the decode worker's first step on
                # this request, so TTFT includes queueing here.
                r.first_token_ms = -1.0
            if not active:
                now = pending[idx].transfer_done_ms
                continue

            step = self.dcost.decode_ms([ctx[r.request_id] for r in active])
            now += step.duration_ms
            res.decode_busy_ms[w] += step.duration_ms
            done: list[ClusterRequest] = []
            for r in active:
                if r.first_token_ms < 0:
                    r.first_token_ms = now
                remaining[r.request_id] -= 1
                ctx[r.request_id] += 1
                if remaining[r.request_id] <= 0:
                    r.finish_ms = now
                    done.append(r)
            for r in done:
                active.remove(r)

    # ------------------------------------------------------------------
    def run_colocated(self, reqs: list[ClusterRequest], n_workers: int) -> ClusterResult:
        """Baseline: every worker does both phases, sharing a token budget."""
        res = ClusterResult("colocated", n_workers, n_workers)
        res.prefill_busy_ms = [0.0] * n_workers
        res.decode_busy_ms = [0.0] * n_workers
        groups: list[list[ClusterRequest]] = [[] for _ in range(n_workers)]
        for i, r in enumerate(sorted(reqs, key=lambda x: x.arrival_ms)):
            groups[i % n_workers].append(r)
        for w, group in enumerate(groups):
            self._colocated_worker(group, res, w)
        res.makespan_ms = max((r.finish_ms for r in reqs if r.finish_ms >= 0),
                              default=0.0)
        res.requests = reqs
        return res

    def _colocated_worker(self, group: list[ClusterRequest], res: ClusterResult,
                          w: int) -> None:
        if not group:
            return
        cfg = self.cfg
        pending = sorted(group, key=lambda r: r.arrival_ms)
        idx = 0
        active: list[ClusterRequest] = []
        prefilling: list[tuple[ClusterRequest, int]] = []   # (req, tokens done)
        remaining: dict[str, int] = {}
        ctx: dict[str, int] = {}
        now = pending[0].arrival_ms

        while idx < len(pending) or active or prefilling:
            while (idx < len(pending) and pending[idx].arrival_ms <= now
                   and len(active) + len(prefilling) < cfg.max_batch):
                r = pending[idx]; idx += 1
                prefilling.append((r, 0))
                r.prefill_start_ms = now
                remaining[r.request_id] = r.output_tokens
                ctx[r.request_id] = r.prompt_tokens
            if not active and not prefilling:
                now = pending[idx].arrival_ms
                continue

            budget = cfg.max_num_batched_tokens
            chunks: list[int] = []
            chunk_ctx: list[int] = []
            newly_done: list[ClusterRequest] = []

            if cfg.chunked_prefill:
                budget -= len(active)      # decodes are admitted first
            # allocate prefill chunks from whatever budget remains
            still: list[tuple[ClusterRequest, int]] = []
            for r, done_tok in prefilling:
                need = r.prompt_tokens - done_tok
                take = min(need, max(budget, 0))
                if take > 0:
                    chunks.append(take)
                    chunk_ctx.append(done_tok + take)
                    budget -= take
                if done_tok + take >= r.prompt_tokens:
                    newly_done.append(r)
                else:
                    still.append((r, done_tok + take))
            prefilling = still

            decode_ctx = ([ctx[r.request_id] for r in active]
                          if (cfg.chunked_prefill or not chunks) else [])
            if chunks and decode_ctx:
                step = self.dcost.mixed_ms(chunks, chunk_ctx, decode_ctx)
            elif chunks:
                step = self.dcost.prefill_ms(chunks, chunk_ctx)
            else:
                step = self.dcost.decode_ms(decode_ctx)
            now += step.duration_ms
            if chunks:
                res.prefill_busy_ms[w] += step.duration_ms
            else:
                res.decode_busy_ms[w] += step.duration_ms

            for r in newly_done:
                r.prefill_done_ms = now
                r.transfer_done_ms = now       # no transfer when colocated
                r.first_token_ms = now
                r.decode_worker = w
                r.prefill_worker = w
                active.append(r)
                ctx[r.request_id] = r.prompt_tokens
                remaining[r.request_id] -= 1
                if remaining[r.request_id] <= 0:
                    r.finish_ms = now

            if decode_ctx:
                finished: list[ClusterRequest] = []
                for r in active:
                    if r in newly_done:
                        continue
                    remaining[r.request_id] -= 1
                    ctx[r.request_id] += 1
                    if remaining[r.request_id] <= 0:
                        r.finish_ms = now
                        finished.append(r)
                for r in finished:
                    active.remove(r)
            for r in [x for x in active if remaining[x.request_id] <= 0]:
                active.remove(r)


def optimal_split(model: str, gpu: str, *, total_gpus: int,
                  input_len: int, output_len: int,
                  kv_dtype: str = "fp16") -> dict[str, Any]:
    """Analytic prefill/decode pool ratio for a steady workload.

    Balance point: prefill work per request is roughly linear in input length,
    decode work is roughly linear in output length times the per-step cost at
    the achieved batch size. The ratio of those two totals is the ratio of pool
    sizes, which is why the right split moves with the workload and a fixed
    50/50 is almost never correct.
    """
    m, g = get_model(model), get_gpu(gpu)
    cm = CostModel(m, g, kv_dtype=kv_dtype)
    prefill_ms = cm.prefill_ms([input_len]).duration_ms
    # Decode cost per request, assuming a healthy batch.
    batch = 64
    per_step = cm.decode_ms([input_len + output_len // 2] * batch).duration_ms
    decode_ms = output_len * per_step / batch

    total = prefill_ms + decode_ms
    p_frac = prefill_ms / total
    n_prefill = max(1, round(total_gpus * p_frac))
    n_decode = max(1, total_gpus - n_prefill)

    link = INTERCONNECTS["rdma-400g"]
    xfer = link.transfer_ms(input_len * m.kv_bytes_per_token(kv_dtype))
    return {
        "prefill_ms_per_request": round(prefill_ms, 2),
        "decode_ms_per_request": round(decode_ms, 2),
        "prefill_fraction": round(p_frac, 3),
        "suggested_prefill_gpus": n_prefill,
        "suggested_decode_gpus": n_decode,
        "kv_transfer_mib": round(input_len * m.kv_bytes_per_token(kv_dtype) / 1024**2, 1),
        "kv_transfer_ms_rdma400": round(xfer, 2),
        "transfer_as_pct_of_prefill": round(xfer / max(prefill_ms, 1e-9) * 100, 1),
        "verdict": (
            "transfer cost is small relative to prefill: disaggregation is viable"
            if xfer < prefill_ms * 0.35 else
            "transfer cost is large relative to prefill: disaggregation will "
            "likely hurt TTFT on this workload"
        ),
    }
