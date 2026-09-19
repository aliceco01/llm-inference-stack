"""Speculative decoding: the arithmetic that decides whether it pays.

The mechanism: a cheap draft model proposes k tokens, the expensive target
model verifies all k in a single forward pass, and a rejection-sampling step
accepts a prefix of them. Because the acceptance test is constructed so the
accepted tokens are distributed exactly as the target would have produced them,
**output quality is unchanged**. That is what makes it the closest thing to a
free lunch in inference: strictly faster decode, identical distribution.

It works because decode is memory-bandwidth bound. Verifying k+1 tokens in one
forward costs barely more than generating one, since the weights are read once
either way. You are spending spare FLOPs to buy back memory traffic.

The arithmetic that gets misreported:

Accepted tokens per iteration is NOT k * alpha. Rejection sampling stops at the
FIRST rejection, so with per-token acceptance alpha the expected number of
accepted draft tokens is alpha + alpha^2 + ... + alpha^k, and one bonus token is
always emitted. Total expected tokens per iteration:

    E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)

Modelling acceptance as k*alpha overstates the gain badly at large k. At
alpha=0.7 with k=8, the naive count says 5.6 draft tokens are accepted; the
true expectation is 2.20, giving 3.20 tokens per iteration rather than 6.6.
The error grows with k precisely where the naive model predicts the biggest
wins, which is why speculative decoding gets oversold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def expected_tokens(alpha: float, k: int) -> float:
    """Expected tokens emitted per target forward pass.

    alpha is the per-token acceptance probability, assumed iid. Real acceptance
    is correlated (a draft that has drifted stays wrong), so this is an upper
    bound on what you will measure.
    """
    if k <= 0:
        return 1.0
    if alpha >= 1.0:
        return k + 1.0
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def iteration_cost(k: int, draft_cost_ratio: float,
                   verify_overhead: float = 0.0) -> float:
    """Cost of one speculative iteration, in units of a plain decode step.

    k sequential draft forwards plus one target verify. The verify processes
    k+1 tokens instead of 1, which is nearly free while decode is
    bandwidth-bound but stops being free once the batch is large enough that
    the verify becomes compute-bound: that is what `verify_overhead` captures.
    """
    return k * draft_cost_ratio + 1.0 + verify_overhead * k


def speedup(alpha: float, k: int, draft_cost_ratio: float,
            verify_overhead: float = 0.0) -> float:
    """Decode speedup versus no speculation."""
    return expected_tokens(alpha, k) / iteration_cost(k, draft_cost_ratio, verify_overhead)


def optimal_k(alpha: float, draft_cost_ratio: float,
              verify_overhead: float = 0.0, k_max: int = 16) -> tuple[int, float]:
    """Best draft length and its speedup.

    There is always an optimum: gains from longer drafts decay geometrically
    (alpha^k) while cost grows linearly (k * c), so past some k every extra
    draft token costs more than it returns.
    """
    best_k, best = 0, 1.0
    for k in range(0, k_max + 1):
        s = speedup(alpha, k, draft_cost_ratio, verify_overhead)
        if s > best:
            best_k, best = k, s
    return best_k, best


def breakeven_alpha(k: int, draft_cost_ratio: float,
                    verify_overhead: float = 0.0) -> float:
    """Acceptance rate below which speculation is a net loss.

    Below this, you are paying for draft forwards that get rejected. Worth
    computing before deploying, and worth monitoring after: acceptance is
    workload dependent and a traffic mix shift can silently push you under it.
    """
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if speedup(mid, k, draft_cost_ratio, verify_overhead) > 1.0:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class DraftCandidate:
    """A draft strategy and its measured or assumed characteristics."""

    name: str
    kind: str                       # "draft_model" | "ngram" | "eagle" | "medusa"
    cost_ratio: float               # draft step cost / target step cost
    typical_alpha: float            # acceptance rate, workload dependent
    extra_vram_gib: float = 0.0
    notes: str = ""


# Representative characteristics. Acceptance rates vary enormously by workload,
# so these are starting points for ranking candidates, not results.
CANDIDATES: list[DraftCandidate] = [
    DraftCandidate(
        "llama-3.2-1b draft", "draft_model", cost_ratio=0.16, typical_alpha=0.72,
        extra_vram_gib=2.5,
        notes="Same tokenizer and family as the 8B/70B target, which matters: a "
              "draft from a different family has a different vocabulary and "
              "cannot be used at all."),
    DraftCandidate(
        "ngram / prompt lookup", "ngram", cost_ratio=0.005, typical_alpha=0.55,
        extra_vram_gib=0.0,
        notes="No model at all: proposes continuations copied from the prompt. "
              "Near-zero cost and near-zero risk. Excellent for summarisation, "
              "RAG, code editing and any task that quotes its input; useless for "
              "open-ended generation."),
    DraftCandidate(
        "EAGLE-2 head", "eagle", cost_ratio=0.06, typical_alpha=0.80,
        extra_vram_gib=1.2,
        notes="A small head trained on the target's own hidden states. Highest "
              "acceptance per unit cost, but needs a head trained for that exact "
              "target checkpoint."),
    DraftCandidate(
        "Medusa heads", "medusa", cost_ratio=0.04, typical_alpha=0.65,
        extra_vram_gib=1.5,
        notes="Multiple decoding heads predicting positions t+1..t+k in parallel. "
              "Cheap, but independent heads make later positions weak."),
]


@dataclass
class SpecPlan:
    candidate: str
    alpha: float
    k: int
    speedup: float
    breakeven_alpha: float
    extra_vram_gib: float
    notes: str = ""
    warnings: list[str] = field(default_factory=list)


def plan(candidates: Sequence[DraftCandidate] | None = None, *,
         alpha_override: float | None = None, k_max: int = 12,
         batch_size: int = 1, verify_overhead: float | None = None) -> list[SpecPlan]:
    """Rank draft strategies.

    `batch_size` matters and is usually ignored. Speculative decoding spends
    FLOPs to save bandwidth, and that trade only works while decode is
    bandwidth-bound. At large batch sizes the target is already compute-bound,
    verifying k+1 tokens per sequence costs real time, and the speedup
    collapses. This is why speculation is a single-user and low-concurrency
    optimisation, and why it can make a saturated server slower.
    """
    cands = list(candidates or CANDIDATES)
    # Verify cost grows with batch: at batch 1 it is nearly free, by batch ~64
    # the extra tokens are real compute.
    if verify_overhead is None:
        verify_overhead = min(0.9, max(0.0, (batch_size - 1) / 64.0 * 0.6))
    out: list[SpecPlan] = []
    for c in cands:
        a = alpha_override if alpha_override is not None else c.typical_alpha
        k, s = optimal_k(a, c.cost_ratio, verify_overhead, k_max=k_max)
        be = breakeven_alpha(max(k, 1), c.cost_ratio, verify_overhead)
        p = SpecPlan(candidate=c.name, alpha=a, k=k, speedup=s,
                     breakeven_alpha=be, extra_vram_gib=c.extra_vram_gib,
                     notes=c.notes)
        if s <= 1.02:
            p.warnings.append(
                f"no useful gain at batch {batch_size}: verification overhead "
                "cancels the benefit. Speculation is a low-concurrency optimisation.")
        if a < be:
            p.warnings.append(
                f"acceptance {a:.2f} is below breakeven {be:.2f}: this would be "
                "slower than plain decoding.")
        if c.extra_vram_gib:
            p.warnings.append(
                f"{c.extra_vram_gib:.1f} GiB of VRAM leaves the KV cache, reducing "
                "max concurrency. Check against project 03 before enabling.")
        out.append(p)
    return sorted(out, key=lambda p: -p.speedup)
