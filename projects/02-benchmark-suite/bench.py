#!/usr/bin/env python3
"""TTFT / ITL / throughput benchmark harness.

    # closed-loop concurrency ladder against any OpenAI-compatible endpoint
    ./bench.py sweep --base-url http://localhost:8000 --model llama-3.1-8b \
        --workload balanced --concurrency 1,2,4,8,16,32,64 --requests-per-point 64

    # open-loop: the mode that finds the capacity cliff
    ./bench.py rate --base-url http://localhost:8000 --model llama-3.1-8b \
        --rps 1,2,5,10,20,40 --duration 30

    # one request, fully instrumented
    ./bench.py single --base-url http://localhost:8000 --model llama-3.1-8b

    # compare two configurations on one chart
    ./bench.py compare --config configs/chunked_prefill.yaml

Produces: results/<run-id>.json, charts, and a markdown report with the
SLO-bounded capacity called out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmkit import (
    SLO,
    EndpointConfig,
    Run,
    RunMeta,
    StreamingClient,
    WorkloadGenerator,
    concurrency_ladder,
    find_knee,
    new_run_id,
    probe,
    report,
    run_closed_loop,
    run_open_loop,
    summarize,
    sweep_table,
)
from llmkit.workload import PRESETS, LengthSpec, WorkloadSpec


# ---------------------------------------------------------------------------
def _plan_point(args: argparse.Namespace, concurrency: int) -> tuple[int, int]:
    """Measured requests and warmup requests for one concurrency level.

    Both scale with concurrency, and that is the point. A closed-loop run at
    concurrency N starts with all N workers firing at once into an empty
    engine, so the first full wave is pure startup transient. Excluding a fixed
    4 requests leaves most of that transient in the sample at high N, which
    produces non-monotonic load curves that look like engine pathology and are
    actually measurement error.

    Rules:
      * warm up at least one full wave (N requests), so every measured request
        starts against an already-busy engine;
      * measure at least 10 waves, so a p95 is computed from enough samples to
        be stable rather than from a handful of transient outliers.
    """
    warm = args.warmup_requests if args.warmup_requests_explicit else max(
        args.warmup_requests, concurrency
    )
    n = args.requests_per_point or max(concurrency * 10, 64)
    return n, warm


def _check_sample_size(summary, n: int, concurrency: int) -> None:
    """Refuse to report a percentile the sample cannot support."""
    waves = n / max(concurrency, 1)
    if waves < 8:
        summary.warnings.append(
            f"only {waves:.1f} request-waves at concurrency {concurrency} "
            f"({n} requests): the sample is dominated by the startup transient "
            "and p95/p99 are not trustworthy. Raise --requests-per-point."
        )
    n_ok = summary.n_ok
    if n_ok < 100:
        # Nearest-rank p99 on 48 samples is just the max: it carries no
        # information about the 99th percentile of the distribution.
        summary.warnings.append(
            f"n={n_ok} successful requests: p99 is effectively the maximum "
            "observation. At least 100 samples are needed for a meaningful p99."
        )


# ---------------------------------------------------------------------------
def make_endpoint(args: argparse.Namespace) -> EndpointConfig:
    return EndpointConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        api=args.api,
        timeout_s=args.timeout,
        supports_ignore_eos=not args.no_ignore_eos,
        supports_min_tokens=not args.no_ignore_eos,
    )


def make_workload(args: argparse.Namespace) -> WorkloadSpec:
    if args.workload in PRESETS:
        spec = PRESETS[args.workload]
    else:
        spec = WorkloadSpec(name="custom")
    # explicit overrides win over the preset
    if args.input_len:
        spec.input_len = LengthSpec("fixed", args.input_len)
    if args.output_len:
        spec.output_len = LengthSpec("fixed", args.output_len)
    if args.system_prompt_tokens is not None:
        spec.system_prompt_tokens = args.system_prompt_tokens
    spec.seed = args.seed
    return spec


async def warm_up(client: StreamingClient, spec: WorkloadSpec, n: int) -> None:
    """Fire throwaway requests so measurements start from a warm engine.

    The first requests to a fresh engine pay CUDA graph capture, lazy kernel
    compilation and an empty prefix cache. Including them turns a cold-start
    artefact into a permanent-looking latency number.
    """
    if n <= 0:
        return
    gen = WorkloadGenerator(spec)
    reqs = list(gen.stream(n))
    await asyncio.gather(*[client.send(r) for r in reqs], return_exceptions=True)


# ---------------------------------------------------------------------------
async def cmd_single(args: argparse.Namespace) -> int:
    ep = make_endpoint(args)
    info = await probe(ep)
    print(f"endpoint: {json.dumps(info)[:300]}\n")
    spec = make_workload(args)
    gen = WorkloadGenerator(spec)
    req = next(iter(gen.stream(1)))
    async with StreamingClient(ep) as c:
        rec = await c.send(req)
    print(f"request_id      {rec.request_id}")
    print(f"status          {rec.status_code}  replica={rec.replica}")
    if rec.error:
        print(f"ERROR           {rec.error}")
        return 1
    print(f"prompt tokens   {rec.prompt_tokens}  (source: {rec.extra.get('token_source')})")
    print(f"output tokens   {rec.output_tokens}")
    print(f"cached prompt   {rec.cached_prompt_tokens}")
    print(f"TTFT            {rec.ttft_ms:.2f} ms")
    print(f"  first frame   {rec.ttft_first_chunk_ms:.2f} ms  "
          f"(delta {rec.ttft_ms - rec.ttft_first_chunk_ms:+.2f} ms: the role-only frame)")
    print(f"E2E             {rec.e2e_ms:.2f} ms")
    print(f"TPOT            {rec.tpot_ms:.2f} ms/token")
    itls = rec.itls_ms
    if itls:
        from llmkit import percentile
        print(f"ITL             p50 {percentile(itls,50):.2f}  p95 {percentile(itls,95):.2f}  "
              f"p99 {percentile(itls,99):.2f}  max {max(itls):.2f} ms")
        print(f"decode rate     {rec.decode_tok_per_s:.1f} tok/s")
    return 0


async def cmd_sweep(args: argparse.Namespace) -> int:
    ep = make_endpoint(args)
    spec = make_workload(args)
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)
    levels = ([int(x) for x in args.concurrency.split(",")] if args.concurrency
              else concurrency_ladder(1, args.max_concurrency))

    info = await probe(ep)
    if not info.get("reachable"):
        print(f"ERROR: endpoint not reachable: {info}", file=sys.stderr)
        return 2

    summaries = []
    all_records = []
    async with StreamingClient(ep) as client:
        await warm_up(client, spec, args.warmup_requests)
        for c in levels:
            gen = WorkloadGenerator(spec)
            n, warm = _plan_point(args, c)
            res = await run_closed_loop(
                client, gen.stream(n + warm),
                concurrency=c, max_requests=n + warm,
            )
            s = summarize(
                res.records, label=args.label or spec.name, concurrency=c, slo=slo,
                warmup_requests=warm,
                meta={"driver": "closed_loop", "max_inflight": res.max_inflight,
                      "requests_per_point": n, "warmup_requests": warm},
            )
            _check_sample_size(s, n, c)
            summaries.append(s)
            all_records.extend(res.records)
            print(s.headline())
            for w in s.warnings:
                print(f"    WARNING: {w}")

    print("\n" + sweep_table(summaries))
    return _finish(args, ep, spec, summaries, all_records, slo, "closed_loop", info)


async def cmd_rate(args: argparse.Namespace) -> int:
    ep = make_endpoint(args)
    spec = make_workload(args)
    slo = SLO(ttft_ms=args.slo_ttft, p_itl_ms=args.slo_itl)
    rates = [float(x) for x in args.rps.split(",")]

    info = await probe(ep)
    if not info.get("reachable"):
        print(f"ERROR: endpoint not reachable: {info}", file=sys.stderr)
        return 2

    summaries, all_records = [], []
    async with StreamingClient(ep) as client:
        await warm_up(client, spec, args.warmup_requests)
        for r in rates:
            gen = WorkloadGenerator(spec)
            res = await run_open_loop(
                client, gen.stream(int(r * args.duration * 3) + 100),
                rps=r, duration_s=args.duration,
            )
            s = summarize(
                res.records, label=args.label or spec.name,
                concurrency=int(round(res.max_inflight)), target_rps=r, slo=slo,
                warmup_s=args.warmup_s,
                meta={"driver": "open_loop", "max_inflight": res.max_inflight,
                      "schedule_lag_p95_ms": res.schedule_lag_ms_p95},
            )
            if res.schedule_lag_ms_p95 > 50:
                s.warnings.append(
                    f"arrival schedule lag p95 = {res.schedule_lag_ms_p95:.0f}ms: the "
                    "load generator could not keep up, so offered load was below target. "
                    "Discard this point or shard the generator."
                )
            summaries.append(s)
            all_records.extend(res.records)
            print(f"  rps={r:<6} {s.headline()}  peak_inflight={res.max_inflight}")
            for w in s.warnings:
                print(f"    WARNING: {w}")

    print("\n" + sweep_table(summaries))
    _print_saturation(summaries)
    return _finish(args, ep, spec, summaries, all_records, slo, "open_loop", info)


def _print_saturation(summaries: Sequence[Any]) -> None:
    """Find where offered load exceeds service capacity."""
    print("\noffered vs achieved rate (divergence = saturation):")
    print(f"  {'target rps':>11}{'achieved':>11}{'ratio':>8}{'peak inflight':>15}")
    for s in summaries:
        if not s.target_rps:
            continue
        ratio = s.req_per_s / s.target_rps
        flag = "  <-- saturated" if ratio < 0.95 else ""
        print(f"  {s.target_rps:>11.1f}{s.req_per_s:>11.2f}{ratio:>8.2f}"
              f"{s.meta.get('max_inflight', 0):>15}{flag}")


def _finish(args, ep, spec, summaries, records, slo, driver, probe_info) -> int:
    out_dir = Path(args.out or "results")
    run_id = new_run_id(args.run_id_prefix or "bench")
    served = ""
    try:
        served = (probe_info.get("body") or {}).get("data", [{}])[0].get("id", "")
    except Exception:
        pass
    token_sources = {r.extra.get("token_source") for r in records if r.ok}
    meta = RunMeta(
        run_id=run_id, project="02-benchmark-suite",
        description=args.description or f"{driver} sweep on {spec.name}",
        engine=args.engine, engine_version=args.engine_version,
        model=ep.model, served_model_id=served, endpoint=ep.base_url,
        workload=spec.name, workload_fingerprint=spec.fingerprint(),
        driver=driver, slo=slo.as_dict(), warmup_s=args.warmup_s,
        token_source=("server_usage" if token_sources == {"server_usage"}
                      else ",".join(sorted(x for x in token_sources if x))),
        simulated=args.simulated,
        engine_flags=json.loads(args.engine_flags) if args.engine_flags else {},
        notes=args.notes,
    )
    run = Run(meta=meta, summaries=list(summaries), records=list(records))
    path = run.save(out_dir / f"{run_id}.json")
    print(f"\nsaved {path}")

    problems = meta.validate()
    if problems:
        print("\npublication gate (project 15 will refuse these):")
        for p in problems:
            print(f"  - {p}")

    if args.charts:
        series = {args.label or spec.name: summaries}
        xkey = "concurrency" if driver == "closed_loop" else "target_rps"
        c1 = report.latency_vs_load(series, out_dir / f"{run_id}-latency.png",
                                    slo=slo, x=xkey, simulated=args.simulated)
        c2 = report.throughput_latency_pareto(series, out_dir / f"{run_id}-pareto.png",
                                              slo=slo, simulated=args.simulated)
        c3 = report.goodput_curve(series, out_dir / f"{run_id}-goodput.png",
                                  simulated=args.simulated)
        print(f"charts: {c1.name}, {c2.name}, {c3.name}")

    md = [f"# Benchmark: {meta.description}", "",
          f"- engine: `{meta.engine}` {meta.engine_version}",
          f"- model: `{meta.model}` (served id: `{served or 'unknown'}`)",
          f"- endpoint: `{meta.endpoint}`",
          f"- workload: `{spec.name}` (fingerprint `{spec.fingerprint()}`)",
          f"- driver: `{driver}`",
          f"- SLO: TTFT <= {slo.ttft_ms:.0f}ms, ITL p{slo.itl_percentile:.0f} <= {slo.p_itl_ms:.0f}ms",
          f"- token counts: {meta.token_source}",
          f"- simulated: **{meta.simulated}**", "",
          report.summaries_md(summaries, slo)]
    if problems:
        md += ["", "## Caveats", ""] + [f"- {p}" for p in problems]
    (out_dir / f"{run_id}.md").write_text("\n".join(md) + "\n")
    print(f"report: {(out_dir / (run_id + '.md'))}")

    knee = find_knee(summaries, slo)
    if knee:
        print(f"\nSLO-bounded capacity: {knee.output_tok_per_s:,.0f} output tok/s "
              f"at concurrency {knee.concurrency}")
    else:
        print("\nNo tested load level met the SLO for 95% of requests.")
    return 0


async def cmd_compare(args: argparse.Namespace) -> int:
    """Run several named endpoint/workload configurations onto one chart."""
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    slo = SLO(**cfg.get("slo", {}))
    out_dir = Path(cfg.get("out", "results"))
    levels = cfg.get("concurrency", [1, 2, 4, 8, 16, 32])
    series: dict[str, list] = {}
    for entry in cfg["targets"]:
        ep = EndpointConfig(
            base_url=entry["base_url"], model=entry["model"],
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        spec = PRESETS.get(entry.get("workload", "balanced"), PRESETS["balanced"])
        name = entry["name"]
        print(f"\n=== {name} ===")
        runs = []
        async with StreamingClient(ep) as client:
            await warm_up(client, spec, cfg.get("warmup_requests", 4))
            for c in levels:
                gen = WorkloadGenerator(spec)
                n = cfg.get("requests_per_point", 48)
                res = await run_closed_loop(client, gen.stream(n), concurrency=c,
                                            max_requests=n)
                s = summarize(res.records, label=name, concurrency=c, slo=slo)
                runs.append(s)
                print("  " + s.headline())
        series[name] = runs
    rid = new_run_id("compare")
    report.latency_vs_load(series, out_dir / f"{rid}-latency.png", slo=slo,
                           simulated=cfg.get("simulated", False))
    report.throughput_latency_pareto(series, out_dir / f"{rid}-pareto.png", slo=slo,
                                     simulated=cfg.get("simulated", False))
    md = [f"# Comparison: {cfg.get('title', rid)}", ""]
    for name, runs in series.items():
        md += [f"## {name}", "", report.summaries_md(runs, slo), ""]
    (out_dir / f"{rid}.md").write_text("\n".join(md))
    print(f"\nwrote {out_dir / (rid + '.md')}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--model", default="llama-3.1-8b")
        p.add_argument("--api", default="chat", choices=["chat", "completions"])
        p.add_argument("--api-key", default=None)
        p.add_argument("--timeout", type=float, default=600.0)
        p.add_argument("--workload", default="balanced",
                       help=f"preset ({', '.join(PRESETS)}) or 'custom'")
        p.add_argument("--input-len", type=int, default=0)
        p.add_argument("--output-len", type=int, default=0)
        p.add_argument("--system-prompt-tokens", type=int, default=None)
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--no-ignore-eos", action="store_true",
                       help="backend lacks ignore_eos/min_tokens; output length "
                            "will vary and throughput becomes less comparable")

    def outputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--slo-ttft", type=float, default=2000.0)
        p.add_argument("--slo-itl", type=float, default=100.0)
        p.add_argument("--out", default="results")
        p.add_argument("--label", default=None)
        p.add_argument("--run-id-prefix", default=None)
        p.add_argument("--description", default=None)
        p.add_argument("--notes", default="")
        p.add_argument("--engine", default="unknown")
        p.add_argument("--engine-version", default="")
        p.add_argument("--engine-flags", default=None, help="JSON dict of serving flags")
        p.add_argument("--simulated", action="store_true",
                       help="mark results as produced by the simulator, not hardware")
        p.add_argument("--charts", action="store_true", default=True)
        p.add_argument("--no-charts", dest="charts", action="store_false")
        p.add_argument("--warmup-requests", type=int, default=4,
                       help="minimum warmup requests; automatically raised to one "
                            "full concurrency wave unless --warmup-requests-exact")
        p.add_argument("--warmup-requests-exact", dest="warmup_requests_explicit",
                       action="store_true", help="use --warmup-requests verbatim")
        p.add_argument("--warmup-s", type=float, default=0.0)

    p = sub.add_parser("single", help="one fully instrumented request")
    common(p)

    p = sub.add_parser("sweep", help="closed-loop concurrency ladder")
    common(p); outputs(p)
    p.add_argument("--concurrency", default=None, help="comma list, e.g. 1,2,4,8")
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--requests-per-point", type=int, default=0)

    p = sub.add_parser("rate", help="open-loop arrival-rate sweep")
    common(p); outputs(p)
    p.add_argument("--rps", default="1,2,5,10,20")
    p.add_argument("--duration", type=float, default=30.0)

    p = sub.add_parser("compare", help="several configs onto one chart")
    p.add_argument("--config", required=True)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    fn = {"single": cmd_single, "sweep": cmd_sweep,
          "rate": cmd_rate, "compare": cmd_compare}[args.cmd]
    return asyncio.run(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
