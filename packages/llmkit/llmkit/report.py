"""Charts and tables.

Plotting rules follow from what the reader has to be able to do with the
figure:

* Latency axes are log scale. Latency spans three orders of magnitude between
  idle and saturated, and a linear axis compresses the entire interesting
  region into the bottom pixel row.
* Every latency curve shows p50 AND p95/p99. A mean-only load curve hides the
  tail, and the tail is the thing that pages you.
* Throughput-vs-latency is drawn as a parametric curve (the Pareto frontier),
  not two separate charts, because the tradeoff is the point.
* SLO lines are drawn explicitly so "does this configuration pass" is a visual
  question rather than an arithmetic one.
* Simulated runs are watermarked. Always.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import RunSummary, find_knee
from .types import SLO

# Colour-blind safe qualitative palette (Okabe-Ito).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#666666"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })
    return plt


def _watermark(fig, simulated: bool) -> None:
    if not simulated:
        return
    fig.text(
        0.5, 0.5, "SIMULATED", fontsize=44, color="#D55E00",
        alpha=0.12, ha="center", va="center", rotation=28, weight="bold", zorder=10,
    )


def latency_vs_load(
    series: dict[str, Sequence[RunSummary]],
    out: str | Path,
    *,
    slo: SLO | None = None,
    x: str = "concurrency",
    title: str = "Latency under rising concurrency",
    simulated: bool = False,
) -> Path:
    """TTFT and ITL percentile curves, one panel each."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, runs) in enumerate(series.items()):
        runs = sorted(runs, key=lambda s: getattr(s, x) or 0)
        xs = [getattr(s, x) for s in runs]
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(xs, [s.ttft.p50 for s in runs], "-o", ms=3.5, color=c, label=f"{name} p50")
        axes[0].plot(xs, [s.ttft.p95 for s in runs], "--^", ms=3.5, color=c, alpha=0.75,
                     label=f"{name} p95")
        axes[1].plot(xs, [s.itl.p50 for s in runs], "-o", ms=3.5, color=c, label=f"{name} p50")
        axes[1].plot(xs, [s.itl.p95 for s in runs], "--^", ms=3.5, color=c, alpha=0.75,
                     label=f"{name} p95")
    if slo:
        axes[0].axhline(slo.ttft_ms, color="#D55E00", ls=":", lw=1.4,
                        label=f"SLO {slo.ttft_ms:.0f}ms")
        axes[1].axhline(slo.p_itl_ms, color="#D55E00", ls=":", lw=1.4,
                        label=f"SLO {slo.p_itl_ms:.0f}ms")
    xlabel = "concurrency" if x == "concurrency" else "offered load (req/s)"
    for ax, t, yl in ((axes[0], "Time to first token", "TTFT (ms)"),
                      (axes[1], "Inter-token latency", "ITL (ms)")):
        ax.set_xscale("log", base=2) if x == "concurrency" else None
        ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, y=1.02, fontsize=11)
    _watermark(fig, simulated)
    return _save(fig, out)


def throughput_latency_pareto(
    series: dict[str, Sequence[RunSummary]],
    out: str | Path,
    *,
    slo: SLO | None = None,
    latency: str = "ttft_p95",
    title: str = "Throughput vs latency",
    simulated: bool = False,
) -> Path:
    """The frontier. Each point is one concurrency level; the knee is marked."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (name, runs) in enumerate(series.items()):
        runs = sorted(runs, key=lambda s: s.concurrency or 0)
        c = PALETTE[i % len(PALETTE)]
        xs = [s.output_tok_per_s for s in runs]
        ys = [_lat(s, latency) for s in runs]
        ax.plot(xs, ys, "-o", ms=4, color=c, label=name)
        for s, xv, yv in zip(runs, xs, ys):
            if s.concurrency:
                ax.annotate(f"{s.concurrency}", (xv, yv), fontsize=6,
                            textcoords="offset points", xytext=(3, 3), color=c)
        if slo:
            knee = find_knee(runs, slo)
            if knee:
                ax.plot([knee.output_tok_per_s], [_lat(knee, latency)], "*",
                        ms=16, color=c, markeredgecolor="black", markeredgewidth=0.4,
                        zorder=5)
    if slo:
        ax.axhline(slo.ttft_ms, color="#D55E00", ls=":", lw=1.4, label="TTFT SLO")
    ax.set_xlabel("output throughput (tok/s)")
    ax.set_ylabel(latency.replace("_", " "))
    ax.set_yscale("log")
    ax.set_title(title + "   (star = SLO-bounded capacity)")
    ax.legend(fontsize=8)
    _watermark(fig, simulated)
    return _save(fig, out)


def goodput_curve(
    series: dict[str, Sequence[RunSummary]],
    out: str | Path,
    *,
    title: str = "Goodput: throughput that actually meets the SLO",
    simulated: bool = False,
) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, (name, runs) in enumerate(series.items()):
        runs = sorted(runs, key=lambda s: s.concurrency or 0)
        c = PALETTE[i % len(PALETTE)]
        xs = [s.concurrency for s in runs]
        ax.plot(xs, [s.output_tok_per_s for s in runs], "--", color=c, alpha=0.45,
                label=f"{name} raw")
        ax.plot(xs, [s.output_tok_per_s * (s.goodput_ratio if not math.isnan(s.goodput_ratio) else 0)
                     for s in runs], "-o", ms=4, color=c, label=f"{name} goodput")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency"); ax.set_ylabel("tok/s")
    ax.set_title(title); ax.legend(fontsize=8)
    _watermark(fig, simulated)
    return _save(fig, out)


def timeline(traces: Sequence[Any], out: str | Path, *, title: str = "Scheduler timeline",
             simulated: bool = True) -> Path:
    """Step-by-step engine behaviour: what ran, and what memory did.

    This is the figure that makes scheduling legible. Prefill steps appear as
    tall spikes in step duration; the decode stall next to them is the cost.
    """
    plt = _mpl()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = [x.t_ms / 1000 for x in traces]
    colors = {"prefill": "#D55E00", "decode": "#0072B2",
              "mixed": "#009E73", "stalled": "#666666"}
    axes[0].scatter(t, [x.dt_ms for x in traces], s=6,
                    c=[colors.get(x.phase, "#999") for x in traces])
    axes[0].set_ylabel("step time (ms)"); axes[0].set_yscale("log")
    axes[0].set_title(title)
    handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k)
               for k, v in colors.items()]
    axes[0].legend(handles=handles, fontsize=7, ncol=4)

    axes[1].plot(t, [x.n_running for x in traces], color="#0072B2", lw=1, label="running")
    axes[1].plot(t, [x.n_waiting for x in traces], color="#E69F00", lw=1, label="waiting")
    axes[1].plot(t, [x.n_swapped for x in traces], color="#CC79A7", lw=1, label="swapped")
    axes[1].set_ylabel("sequences"); axes[1].legend(fontsize=7)

    axes[2].plot(t, [x.kv_util * 100 for x in traces], color="#009E73", lw=1)
    pre = [(x.t_ms / 1000) for x in traces if x.preemptions]
    for p in pre[:400]:
        axes[2].axvline(p, color="#D55E00", alpha=0.25, lw=0.7)
    axes[2].set_ylabel("KV util (%)"); axes[2].set_xlabel("simulated time (s)")
    axes[2].set_ylim(0, 105)
    if pre:
        axes[2].set_title(f"{len(pre)} steps with preemption (orange)", fontsize=8)
    _watermark(fig, simulated)
    return _save(fig, out)


def bar_compare(labels: Sequence[str], groups: dict[str, Sequence[float]], out: str | Path,
                *, ylabel: str = "", title: str = "", simulated: bool = False,
                log: bool = False, annotate: bool = True) -> Path:
    plt = _mpl()
    n = len(groups)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(6.2, 1.3 * len(labels) + 2), 4.0))
    for i, (name, vals) in enumerate(groups.items()):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(labels))]
        b = ax.bar(xs, vals, width=width, label=name, color=PALETTE[i % len(PALETTE)])
        if annotate:
            ax.bar_label(b, fmt="%.4g", fontsize=6.5, padding=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if log:
        ax.set_yscale("log")
    if n > 1:
        ax.legend(fontsize=8)
    _watermark(fig, simulated)
    return _save(fig, out)


def _lat(s: RunSummary, key: str) -> float:
    field, pct = key.rsplit("_", 1)
    return getattr(getattr(s, field), pct)


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


# --- markdown -------------------------------------------------------------
def md_table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_no data_"
    cols = list(columns or rows[0].keys())
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "n/a"
            return f"{v:,.2f}" if abs(v) < 1e5 else f"{v:,.0f}"
        return str(v)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def summaries_md(summaries: Sequence[RunSummary], slo: SLO | None = None) -> str:
    rows = [{
        "label": s.label,
        "conc": s.concurrency,
        "req/s": s.req_per_s,
        "TTFT p50": s.ttft.p50,
        "TTFT p95": s.ttft.p95,
        "TTFT p99": s.ttft.p99,
        "ITL p50": s.itl.p50,
        "ITL p95": s.itl.p95,
        "out tok/s": s.output_tok_per_s,
        "goodput %": s.goodput_ratio * 100 if not math.isnan(s.goodput_ratio) else float("nan"),
        "err": s.n_error,
    } for s in summaries]
    md = md_table(rows)
    if slo:
        knee = find_knee(summaries, slo)
        if knee:
            md += (f"\n\n**SLO-bounded capacity**: {knee.output_tok_per_s:,.0f} output tok/s "
                   f"at concurrency {knee.concurrency} "
                   f"(TTFT p95 {knee.ttft.p95:.0f}ms, ITL p95 {knee.itl.p95:.1f}ms).")
        else:
            md += "\n\n**No tested concurrency met the SLO for 95% of requests.**"
    warn = {w for s in summaries for w in s.warnings}
    if warn:
        md += "\n\n" + "\n".join(f"> WARNING: {w}" for w in sorted(warn))
    return md
