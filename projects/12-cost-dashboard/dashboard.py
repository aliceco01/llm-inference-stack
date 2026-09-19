#!/usr/bin/env python3
"""Cost-per-token dashboard: per-tenant accounting, $/M tokens, MFU and MBU.

    # analyse a benchmark run's records
    ./dashboard.py report --run ../../results/demo-*.records.parquet \
        --gpu h100-sxm --model llama-3.1-8b

    # utilization analysis from token counts
    ./dashboard.py mfu --model llama-3.1-8b --gpu h100-sxm \
        --prompt-tokens 5000000 --output-tokens 800000 --window-s 3600

    # is self-hosting actually cheaper?
    ./dashboard.py breakeven --gpu h100-sxm --model llama-3.1-8b \
        --achieved-tok-s 2500 --api-price 0.60

    # live dashboard scraping engine /metrics
    ./dashboard.py serve --endpoints http://localhost:8000 --port 9100
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmkit import get_gpu, get_model, report
from llmkit.costs import (
    CostModelConfig,
    attribute_costs,
    breakeven_vs_api,
    compute_utilization,
    price_for,
)
from llmkit.prom import CounterRate, scrape_sync
from llmkit.types import FinishReason, RequestRecord


# ---------------------------------------------------------------------------
def _load_records(path: str) -> list[RequestRecord]:
    """Rehydrate RequestRecords from a project 02 result file."""
    p = Path(path)
    rows: list[dict[str, Any]] = []
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(p).to_pylist()
    elif p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    else:
        raise SystemExit(f"unsupported records file: {p}")
    out: list[RequestRecord] = []
    for r in rows:
        rec = RequestRecord(
            request_id=r.get("request_id", ""),
            model=r.get("model"), tenant=r.get("tenant"),
            session_id=r.get("session_id"),
            prompt_tokens=int(r.get("prompt_tokens") or 0),
            output_tokens=int(r.get("output_tokens") or 0),
            cached_prompt_tokens=int(r.get("cached_prompt_tokens") or 0),
            t_send_ns=int(r.get("t_send_ns") or -1),
            t_done_ns=int(r.get("t_done_ns") or -1),
        )
        if not r.get("ok", True):
            rec.error = r.get("error") or "error"
            rec.finish_reason = FinishReason.ERROR
        out.append(rec)
    return out


def cmd_report(args) -> int:
    recs = _load_records(args.run)
    if not recs:
        raise SystemExit("no records found")
    t0 = min((r.t_send_ns for r in recs if r.t_send_ns >= 0), default=0)
    t1 = max((r.t_done_ns for r in recs if r.t_done_ns >= 0), default=t0)
    window_s = max((t1 - t0) / 1e9, 1e-9)

    price = price_for(args.gpu, n_gpus=args.n_gpus,
                      usd_per_gpu_hour=args.usd_per_gpu_hour,
                      overhead_multiplier=args.overhead)
    cfg = CostModelConfig(weight_prompt=args.weight_prompt,
                          weight_cached_prompt=args.weight_cached_prompt,
                          weight_output=args.weight_output)
    by, fleet = attribute_costs(recs, price, window_s=window_s, cfg=cfg)

    print(f"window {window_s:.1f}s on {price.n_gpus}x {price.gpu} "
          f"at ${price.usd_per_hour:.2f}/hour (incl. {args.overhead:.2f}x overhead)\n")
    rows = [u.as_dict() for u in sorted(by.values(), key=lambda x: -x.usd)]
    print(report.md_table(rows, columns=[
        "tenant", "model", "requests", "prompt_tokens", "output_tokens",
        "cache_hit_rate", "usd", "usd_per_m_output"]))
    print(f"\nfleet: {json.dumps(fleet, indent=2)}")

    mean_ctx = (sum(r.prompt_tokens for r in recs) / max(len(recs), 1)
                + sum(r.output_tokens for r in recs) / max(len(recs), 1) / 2)
    u = compute_utilization(
        args.model, args.gpu, window_s=window_s,
        prompt_tokens=fleet["prompt_tokens"], output_tokens=fleet["output_tokens"],
        cached_prompt_tokens=fleet["cached_prompt_tokens"],
        mean_context=mean_ctx, mean_batch=args.mean_batch, n_gpus=args.n_gpus)
    _print_utilization(u)

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "cost-report.json").write_text(json.dumps(
            {"fleet": fleet, "tenants": rows,
             "utilization": {"prefill_mfu": u.prefill_mfu,
                             "decode_mfu": u.decode_mfu,
                             "decode_mbu": u.decode_mbu}}, indent=2, default=str))
        if rows:
            report.bar_compare(
                [r["tenant"] for r in rows],
                {"$ / M output tokens": [r["usd_per_m_output"] for r in rows]},
                out / "cost-per-tenant.png", ylabel="USD / M tokens",
                title="Cost per million output tokens, by tenant")
        print(f"\nwrote {out}/cost-report.json")
    return 0


def _print_utilization(u) -> None:
    print("\nutilization")
    print(f"  prefill MFU    {u.prefill_mfu*100:>7.2f}%   (compute bound: this is "
          "the number to optimise)")
    print(f"  decode  MBU    {u.decode_mbu*100:>7.2f}%   (bandwidth bound: this is "
          "the number to optimise)")
    print(f"  decode  MFU    {u.decode_mfu*100:>7.2f}%   (structurally low; not a defect)")
    print(f"  blended MFU    {u.blended_mfu*100:>7.2f}%   (reported because people ask; "
          "it hides which phase is inefficient)")
    for v in u.verdict():
        print(f"    - {v}")


def cmd_mfu(args) -> int:
    u = compute_utilization(
        args.model, args.gpu, window_s=args.window_s,
        prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
        cached_prompt_tokens=args.cached_prompt_tokens,
        mean_context=args.mean_context, mean_batch=args.mean_batch,
        n_gpus=args.n_gpus)
    g = get_gpu(args.gpu)
    m = get_model(args.model)
    print(f"{m.describe()}\non {args.n_gpus}x {g.name} "
          f"(peak {g.bf16_tflops:.0f} TFLOP/s, {g.bandwidth_gb_s:.0f} GB/s)\n")
    print(f"  window              {args.window_s:,.0f} s")
    print(f"  prompt tokens       {args.prompt_tokens:,} "
          f"({args.cached_prompt_tokens:,} cached, not prefilled)")
    print(f"  output tokens       {args.output_tokens:,}")
    print(f"  mean batch          {args.mean_batch}")
    _print_utilization(u)

    price = price_for(args.gpu, n_gpus=args.n_gpus,
                      usd_per_gpu_hour=args.usd_per_gpu_hour,
                      overhead_multiplier=args.overhead)
    cost = price.usd_per_second * args.window_s
    print("\ncost")
    print(f"  window cost         ${cost:,.2f}")
    print(f"  $ / M output tokens ${cost / max(args.output_tokens,1) * 1e6:,.3f}")
    print(f"  $ / M total tokens  "
          f"${cost / max(args.output_tokens + args.prompt_tokens,1) * 1e6:,.3f}")
    print("""
Why both metrics: decode reads the full weight matrix every step to produce one
token per sequence, so its arithmetic intensity is tiny and its MFU is capped by
physics at a few percent. Chasing decode MFU is chasing a number you cannot
move. MBU is what responds to batch size, quantization and KV dtype.""")
    return 0


def cmd_breakeven(args) -> int:
    price = price_for(args.gpu, n_gpus=args.n_gpus,
                      usd_per_gpu_hour=args.usd_per_gpu_hour,
                      overhead_multiplier=args.overhead)
    per_m = price.usd_per_hour / max(args.achieved_tok_s * 3600, 1) * 1e6
    r = breakeven_vs_api(per_m, args.api_price)
    print(f"{args.n_gpus}x {price.gpu} at ${price.usd_per_hour:.2f}/hour "
          f"sustaining {args.achieved_tok_s:,.0f} output tok/s\n")
    print(f"  self-hosted   ${per_m:,.3f} / M output tokens (at 100% utilisation)")
    print(f"  API           ${args.api_price:,.3f} / M output tokens")
    print(f"  ratio         {r['ratio']:.3f}")
    print(f"\n  {r['verdict']}")
    print(f"""
  The comparison people get wrong: that self-hosted figure assumes the GPU is
  busy 100% of the time. You pay for it whether or not it is. At {r['min_utilization_to_break_even']*100:.0f}%
  average utilisation the two are equal, and below that the API wins.

  So the real question is not "is the per-token cost lower" but "can we keep
  the fleet above {r['min_utilization_to_break_even']*100:.0f}% utilisation", which is an autoscaling and
  traffic-shaping question (project 11), not a hardware question.

  Also excluded here and easy to forget: engineering time, on-call, model
  update cycles, and the capacity you must hold in reserve for peak.""")
    return 0


# ---------------------------------------------------------------------------
DASH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Inference cost dashboard</title>
<meta http-equiv="refresh" content="10">
<style>
 :root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8b93a7;--acc:#0072B2;--warn:#E69F00;--bad:#D55E00;--ok:#009E73}
 @media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#14171f;--mut:#5b6377}}
 body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:24px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:var(--mut);font-size:12px;margin-bottom:20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:24px}
 .card{border:1px solid #3a3f4b33;border-radius:10px;padding:14px}
 .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
 .v{font-size:24px;font-weight:600;margin-top:4px}
 .u{font-size:12px;color:var(--mut)}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:right;padding:7px 10px;border-bottom:1px solid #3a3f4b33}
 th:first-child,td:first-child{text-align:left}
 th{color:var(--mut);font-weight:500;font-size:11px;text-transform:uppercase}
 .note{color:var(--mut);font-size:12px;margin-top:18px;max-width:70ch}
 .bar{height:6px;background:#3a3f4b33;border-radius:3px;overflow:hidden;margin-top:6px}
 .bar>i{display:block;height:100%;background:var(--acc)}
</style></head><body>
<h1>Inference cost dashboard</h1>
<div class="sub">%(subtitle)s</div>
<div class="grid">%(cards)s</div>
<h1>Per tenant</h1><div class="sub">weighted attribution of shared capacity</div>
%(table)s
<div class="note">%(note)s</div>
</body></html>"""


def _card(k: str, v: str, u: str = "", pct: float | None = None) -> str:
    bar = (f'<div class="bar"><i style="width:{min(max(pct,0),1)*100:.0f}%"></i></div>'
           if pct is not None else "")
    return f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>' \
           f'<div class="u">{u}</div>{bar}</div>'


@dataclass
class LiveState:
    started: float = field(default_factory=time.time)
    prompt: float = 0.0
    output: float = 0.0
    cached: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    rates: dict[str, CounterRate] = field(default_factory=dict)


def cmd_serve(args) -> int:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    price = price_for(args.gpu, n_gpus=args.n_gpus * len(endpoints),
                      usd_per_gpu_hour=args.usd_per_gpu_hour,
                      overhead_multiplier=args.overhead)
    state = LiveState()
    app = FastAPI(title="cost dashboard")

    def poll() -> dict[str, Any]:
        prompt = output = cached = 0.0
        running = waiting = 0.0
        for ep in endpoints:
            m = scrape_sync(ep)
            prompt += m.get("vllm:prompt_tokens_total", 0.0)
            output += m.get("vllm:generation_tokens_total", 0.0)
            cached += m.get("vllm:prefix_cache_hits_total", 0.0)
            running += m.get("vllm:num_requests_running", 0.0)
            waiting += m.get("vllm:num_requests_waiting", 0.0)
        state.prompt, state.output, state.cached = prompt, output, cached
        now = time.time()
        for name, val in (("prompt", prompt), ("output", output)):
            state.rates.setdefault(name, CounterRate(name)).update(val, now)
        return {"prompt": prompt, "output": output, "cached": cached,
                "running": running, "waiting": waiting}

    @app.get("/api/stats")
    async def api_stats() -> JSONResponse:
        s = poll()
        elapsed = max(time.time() - state.started, 1e-9)
        cost = price.usd_per_second * elapsed
        u = compute_utilization(
            args.model, args.gpu, window_s=elapsed,
            prompt_tokens=int(s["prompt"]), output_tokens=int(s["output"]),
            cached_prompt_tokens=int(s["cached"]),
            mean_context=args.mean_context, mean_batch=max(s["running"], 1),
            n_gpus=args.n_gpus * len(endpoints))
        return JSONResponse({
            **s, "elapsed_s": elapsed, "usd": cost,
            "usd_per_m_output": cost / max(s["output"], 1) * 1e6,
            "output_tok_per_s": state.rates["output"].rate,
            "prompt_tok_per_s": state.rates["prompt"].rate,
            "prefill_mfu": u.prefill_mfu, "decode_mbu": u.decode_mbu,
            "decode_mfu": u.decode_mfu,
        })

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        s = poll()
        elapsed = max(time.time() - state.started, 1e-9)
        cost = price.usd_per_second * elapsed
        out_rate = state.rates.get("output", CounterRate("output")).rate
        u = compute_utilization(
            args.model, args.gpu, window_s=elapsed,
            prompt_tokens=int(s["prompt"]), output_tokens=int(s["output"]),
            cached_prompt_tokens=int(s["cached"]),
            mean_context=args.mean_context, mean_batch=max(s["running"], 1),
            n_gpus=args.n_gpus * len(endpoints))
        per_m = cost / max(s["output"], 1) * 1e6
        cards = "".join([
            _card("spend so far", f"${cost:,.2f}",
                  f"${price.usd_per_hour:,.2f}/hour, {price.n_gpus} GPU"),
            _card("$ / M output tokens", f"${per_m:,.2f}",
                  "lower is better; falls as utilisation rises"),
            _card("output tok/s", f"{out_rate:,.0f}",
                  f"{s['running']:.0f} running, {s['waiting']:.0f} waiting"),
            _card("prefix cache", f"{s['cached']/max(s['prompt'],1)*100:,.1f}%",
                  "prompt tokens served from cache",
                  pct=s['cached'] / max(s['prompt'], 1)),
            _card("prefill MFU", f"{u.prefill_mfu*100:,.1f}%",
                  "compute bound: optimise this", pct=u.prefill_mfu),
            _card("decode MBU", f"{u.decode_mbu*100:,.1f}%",
                  "bandwidth bound: optimise this", pct=u.decode_mbu),
            _card("decode MFU", f"{u.decode_mfu*100:,.2f}%",
                  "structurally low, not a defect"),
        ])
        rows = ("<tr><td>all</td>"
                f"<td>{s['prompt']:,.0f}</td><td>{s['cached']:,.0f}</td>"
                f"<td>{s['output']:,.0f}</td><td>${cost:,.2f}</td>"
                f"<td>${per_m:,.2f}</td></tr>")
        table = ("<table><tr><th>tenant</th><th>prompt tok</th><th>cached</th>"
                 "<th>output tok</th><th>spend</th><th>$/M out</th></tr>"
                 f"{rows}</table>")
        note = ("Per-tenant attribution requires tenant labels on requests; this "
                "live view aggregates the fleet. Use <code>./dashboard.py report</code> "
                "on a benchmark run for the per-tenant split. "
                "decode MFU is reported only to show it is structurally low: decode "
                "reads the whole weight matrix per step to emit one token per "
                "sequence, so a few percent is physics, not misconfiguration.")
        return HTMLResponse(DASH_HTML % {
            "subtitle": f"{', '.join(endpoints)} &middot; {args.model} on "
                        f"{price.n_gpus}x {price.gpu} &middot; refreshes every 10s",
            "cards": cards, "table": table, "note": note})

    print(f"dashboard on http://{args.host}:{args.port}  "
          f"scraping {len(endpoints)} endpoint(s)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def pricing(p):
        p.add_argument("--gpu", default="h100-sxm")
        p.add_argument("--model", default="llama-3.1-8b")
        p.add_argument("--n-gpus", type=int, default=1)
        p.add_argument("--usd-per-gpu-hour", type=float, default=None)
        p.add_argument("--overhead", type=float, default=1.35)
        p.add_argument("--mean-batch", type=float, default=32.0)
        p.add_argument("--mean-context", type=float, default=2048.0)

    p = sub.add_parser("report", help="cost breakdown from a run"); pricing(p)
    p.add_argument("--run", required=True, help=".records.parquet or .jsonl")
    p.add_argument("--weight-prompt", type=float, default=1.0)
    p.add_argument("--weight-cached-prompt", type=float, default=0.1)
    p.add_argument("--weight-output", type=float, default=4.0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("mfu", help="utilization from token counts"); pricing(p)
    p.add_argument("--window-s", type=float, required=True)
    p.add_argument("--prompt-tokens", type=int, required=True)
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--cached-prompt-tokens", type=int, default=0)

    p = sub.add_parser("breakeven", help="self-hosted vs API"); pricing(p)
    p.add_argument("--achieved-tok-s", type=float, required=True)
    p.add_argument("--api-price", type=float, required=True,
                   help="API $/M output tokens")

    p = sub.add_parser("serve", help="live HTML dashboard"); pricing(p)
    p.add_argument("--endpoints", default="http://127.0.0.1:8000")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9100)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return {"report": cmd_report, "mfu": cmd_mfu, "breakeven": cmd_breakeven,
            "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
