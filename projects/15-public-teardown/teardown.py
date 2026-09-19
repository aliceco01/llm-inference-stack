#!/usr/bin/env python3
"""Public benchmark teardown: measure three serving configs, publish honestly.

    ./teardown.py run     --config configs/teardown.yaml
    ./teardown.py publish --results results/ --out site/
    ./teardown.py verify  --results results/      # the publication gate

The distinguishing feature of this project is `verify`. Every run must carry
enough provenance to be reproduced and challenged, and anything that fails the
gate is either excluded or published with the caveat attached in the report
body rather than in a footnote nobody reads.

What the gate requires:
  * a git SHA, on a clean working tree
  * the model the server ACTUALLY reported serving, not the one you intended
  * hardware identity, including whether a GPU was present at all
  * server-reported token counts, not estimates
  * the exact workload fingerprint and SLO
  * an explicit simulated/measured flag

A benchmark missing any of these is an anecdote. Publishing it as a result is
how the field ended up full of numbers nobody can reproduce.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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
)
from llmkit.costs import price_for
from llmkit.metrics import RunSummary
from llmkit.workload import PRESETS, LengthSpec, WorkloadSpec


# ---------------------------------------------------------------------------
def _spec_from(entry: dict[str, Any]) -> WorkloadSpec:
    if isinstance(entry.get("workload"), str):
        return PRESETS[entry["workload"]]
    w = entry.get("workload") or {}
    return WorkloadSpec(
        name=w.get("name", "custom"),
        input_len=LengthSpec(**w.get("input_len", {"kind": "fixed", "value": 1024})),
        output_len=LengthSpec(**w.get("output_len", {"kind": "fixed", "value": 256})),
        system_prompt_tokens=w.get("system_prompt_tokens", 0),
        n_prefix_variants=w.get("n_prefix_variants", 1),
        multi_turn=w.get("multi_turn", 1),
        seed=w.get("seed", 1234),
    )


async def bench_config(entry: dict[str, Any], cfg: dict[str, Any],
                       out_dir: Path) -> Run | None:
    """Run the full sweep for one configuration and save it with provenance."""
    ep = EndpointConfig(base_url=entry["base_url"], model=entry["model"])
    slo = SLO(**cfg.get("slo", {}))
    spec = _spec_from(entry)
    levels = cfg.get("concurrency", concurrency_ladder(1, 64))

    info = await probe(ep)
    if not info.get("reachable"):
        print(f"  SKIP {entry['name']}: endpoint unreachable ({info.get('last_error')})")
        return None
    served = ""
    try:
        served = (info.get("body") or {}).get("data", [{}])[0].get("id", "")
    except Exception:
        pass
    if served and entry.get("expect_model") and served != entry["expect_model"]:
        print(f"  WARNING {entry['name']}: server reports serving {served!r}, "
              f"expected {entry['expect_model']!r}. Recording what it reported.")

    summaries: list[RunSummary] = []
    records = []
    async with StreamingClient(ep) as client:
        # Warm up: CUDA graph capture, lazy kernels, and an empty prefix cache
        # all make the first requests unrepresentative.
        gen = WorkloadGenerator(spec)
        warm = list(gen.stream(cfg.get("warmup_requests", 16)))
        await asyncio.gather(*[client.send(r) for r in warm], return_exceptions=True)

        for c in levels:
            gen = WorkloadGenerator(spec)
            n = max(c * cfg.get("waves", 10), cfg.get("min_requests", 64))
            res = await run_closed_loop(client, gen.stream(n + c),
                                        concurrency=c, max_requests=n + c)
            s = summarize(res.records, label=entry["name"], concurrency=c,
                          slo=slo, warmup_requests=c,
                          meta={"driver": "closed_loop",
                                "requests": n, "waves": n / max(c, 1)})
            summaries.append(s)
            records.extend(res.records)
            print(f"    {s.headline()}")
            for w in s.warnings:
                print(f"      WARNING: {w}")

        # Open loop at the SLO-bounded rate, because a capacity claim from
        # closed-loop data alone is not a capacity claim.
        knee = find_knee(summaries, slo)
        if knee and cfg.get("open_loop", True):
            target = max(knee.req_per_s, 1.0)
            for mult in cfg.get("rate_multipliers", [0.8, 1.0, 1.2]):
                gen = WorkloadGenerator(spec)
                rps = target * mult
                res = await run_open_loop(
                    client, gen.stream(int(rps * cfg.get("open_loop_s", 30)) + 200),
                    rps=rps, duration_s=cfg.get("open_loop_s", 30))
                s = summarize(res.records, label=f"{entry['name']} open",
                              concurrency=res.max_inflight, target_rps=rps, slo=slo,
                              meta={"driver": "open_loop",
                                    "schedule_lag_p95_ms": res.schedule_lag_ms_p95})
                if res.schedule_lag_ms_p95 > 50:
                    s.warnings.append(
                        f"client fell behind its arrival schedule by "
                        f"{res.schedule_lag_ms_p95:.0f}ms at p95: offered load was "
                        "below target, so this point must not be published")
                summaries.append(s)
                records.extend(res.records)
                print(f"    open-loop rps={rps:6.1f}  {s.headline()}")

    token_sources = {r.extra.get("token_source") for r in records if r.ok}
    meta = RunMeta(
        run_id=new_run_id(entry["name"].replace(" ", "-").lower()),
        project="15-public-teardown",
        description=entry.get("description", entry["name"]),
        engine=entry.get("engine", "unknown"),
        engine_version=entry.get("engine_version", ""),
        model=entry["model"], served_model_id=served, endpoint=ep.base_url,
        engine_flags=entry.get("flags", {}),
        workload=spec.name, workload_fingerprint=spec.fingerprint(),
        driver="closed_loop+open_loop", slo=slo.as_dict(),
        token_source=("server_usage" if token_sources == {"server_usage"}
                      else ",".join(sorted(x for x in token_sources if x))),
        simulated=entry.get("simulated", False),
        notes=entry.get("notes", ""),
    )
    run = Run(meta=meta, summaries=summaries, records=records)
    path = run.save(out_dir / f"{meta.run_id}.json")
    print(f"    saved {path.name}")
    return run


async def cmd_run(args) -> int:
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg.get("out", "results")); out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[Run] = []
    for entry in cfg["configs"]:
        print(f"\n=== {entry['name']} ===")
        r = await bench_config(entry, cfg, out_dir)
        if r:
            runs.append(r)
    if not runs:
        print("\nno configurations produced results")
        return 1
    print(f"\n{len(runs)} configuration(s) measured. Next: ./teardown.py verify "
          f"--results {out_dir}")
    return 0


# ---------------------------------------------------------------------------
def _load_runs(results: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(results.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "meta" in d and "summaries" in d:
            d["_path"] = str(p)
            out.append(d)
    return out


def cmd_verify(args) -> int:
    """The publication gate."""
    runs = _load_runs(Path(args.results))
    if not runs:
        print(f"no runs found in {args.results}")
        return 1
    print(f"checking {len(runs)} run(s)\n")
    blocked = 0
    for d in runs:
        meta = RunMeta(**{k: v for k, v in d["meta"].items()
                          if k in RunMeta.__dataclass_fields__})
        problems = meta.validate()
        # Additional publication-specific checks beyond RunMeta.validate.
        sums = d.get("summaries") or []
        if not sums:
            problems.append("no summaries")
        for s in sums:
            for w in s.get("warnings", []):
                problems.append(f"summary warning: {w}")
            if s.get("n_ok", 0) < 100 and not math.isnan(s.get("ttft", {}).get("p99", math.nan)):
                problems.append(
                    f"c={s.get('concurrency')}: p99 reported from {s.get('n_ok')} "
                    "samples (nearest-rank p99 is just the maximum below ~100)")
        status = "BLOCKED" if problems else "OK"
        if problems:
            blocked += 1
        print(f"  [{status}] {meta.run_id}  engine={meta.engine} "
              f"model={meta.model} simulated={meta.simulated}")
        for p in dict.fromkeys(problems):
            print(f"      - {p}")
    print(f"\n{len(runs) - blocked} publishable, {blocked} blocked.")
    if blocked and not args.allow_caveats:
        print("\nRe-run with --allow-caveats to publish these with the caveats")
        print("printed in the report body. Do not publish them silently.")
    return 1 if blocked and not args.allow_caveats else 0


# ---------------------------------------------------------------------------
def cmd_publish(args) -> int:
    results = Path(args.results)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(results)
    if not runs:
        print(f"no runs in {results}")
        return 1

    series: dict[str, list[RunSummary]] = {}
    caveats: dict[str, list[str]] = {}
    simulated_any = False
    for d in runs:
        meta = RunMeta(**{k: v for k, v in d["meta"].items()
                          if k in RunMeta.__dataclass_fields__})
        simulated_any |= bool(meta.simulated)
        name = meta.description or meta.run_id
        closed = [s for s in d["summaries"]
                  if (s.get("meta") or {}).get("driver") != "open_loop"]
        series[name] = [_to_summary(s) for s in closed]
        probs = meta.validate()
        for s in d["summaries"]:
            probs.extend(s.get("warnings") or [])
        if probs:
            caveats[name] = list(dict.fromkeys(probs))

    slo_d = (runs[0]["meta"].get("slo") or {})
    slo = SLO(**{k: v for k, v in slo_d.items() if k in SLO.__dataclass_fields__})

    report.latency_vs_load(series, out / "latency-vs-load.png", slo=slo,
                           simulated=simulated_any)
    report.throughput_latency_pareto(series, out / "throughput-latency.png",
                                     slo=slo, simulated=simulated_any)
    report.goodput_curve(series, out / "goodput.png", simulated=simulated_any)

    price = price_for(args.gpu, n_gpus=args.n_gpus)
    cost_rows = []
    for name, sums in series.items():
        knee = find_knee(sums, slo)
        if not knee:
            cost_rows.append({"config": name, "SLO-bounded tok/s": float("nan"),
                              "$/M output tokens": float("nan"),
                              "note": "no tested load met the SLO"})
            continue
        per_m = price.usd_per_hour / max(knee.output_tok_per_s * 3600, 1) * 1e6
        cost_rows.append({
            "config": name,
            "concurrency at knee": knee.concurrency,
            "SLO-bounded tok/s": knee.output_tok_per_s,
            "TTFT p95 (ms)": knee.ttft.p95,
            "ITL p95 (ms)": knee.itl.p95,
            "$/M output tokens": per_m,
        })

    md = _report_md(runs, series, cost_rows, caveats, slo, price,
                    simulated_any, args)
    (out / "README.md").write_text(md)
    for p in ("latency-vs-load.png", "throughput-latency.png", "goodput.png"):
        if not (out / p).exists():
            continue
    print(f"wrote {out}/README.md and charts")
    if caveats:
        print(f"\n{len(caveats)} configuration(s) carry caveats, included in the "
              "report body.")
    return 0


def _to_summary(d: dict[str, Any]) -> RunSummary:
    from llmkit.metrics import Dist
    s = RunSummary()
    for k, v in d.items():
        if k in ("ttft", "itl", "itl_per_req_p95", "tpot", "e2e", "queue"):
            setattr(s, k, Dist(**v))
        elif k in RunSummary.__dataclass_fields__:
            setattr(s, k, v)
    return s


def _report_md(runs, series, cost_rows, caveats, slo, price, simulated, args) -> str:
    lines = [
        "# Serving configuration teardown", "",
    ]
    if simulated:
        lines += [
            "> **These numbers are SIMULATED.** They come from a roofline model "
            "of the engine, not from hardware. They are published to demonstrate "
            "the methodology and are not a claim about any real deployment. Every "
            "chart is watermarked accordingly.", "",
        ]
    lines += [
        "## What was measured", "",
        f"- SLO: TTFT <= {slo.ttft_ms:.0f} ms, ITL p{slo.itl_percentile:.0f} "
        f"<= {slo.p_itl_ms:.0f} ms",
        "- Capacity is reported as the highest concurrency where **95% of "
        "requests meet both**, not as peak throughput",
        f"- Cost basis: {price.n_gpus}x {price.gpu} at "
        f"${price.usd_per_hour:.2f}/hour including a "
        f"{price.overhead_multiplier:.2f}x infrastructure overhead", "",
        "## Configurations", "",
    ]
    rows = []
    for d in runs:
        m = d["meta"]
        rows.append({
            "config": m.get("description") or m.get("run_id"),
            "engine": f"{m.get('engine')} {m.get('engine_version')}".strip(),
            "model served": m.get("served_model_id") or m.get("model"),
            "hardware": ", ".join(m.get("host", {}).get("gpus") or ["no GPU detected"]),
            "flags": ", ".join(f"{k}={v}" for k, v in (m.get("engine_flags") or {}).items()) or "-",
            "tokens": m.get("token_source"),
            "simulated": m.get("simulated"),
        })
    lines += [report.md_table(rows), "", "## Results", "",
              report.md_table(cost_rows), "",
              "![latency vs load](latency-vs-load.png)", "",
              "![throughput vs latency](throughput-latency.png)", "",
              "![goodput](goodput.png)", ""]

    lines += ["## Full sweep", ""]
    for name, sums in series.items():
        lines += [f"### {name}", "", report.summaries_md(sums, slo), ""]

    if caveats:
        lines += ["## Caveats", "",
                  "These are limitations of the measurements above, listed here "
                  "rather than in a footnote because they change how the numbers "
                  "should be read.", ""]
        for name, cs in caveats.items():
            lines += [f"**{name}**", ""]
            lines += [f"- {c}" for c in cs]
            lines += [""]

    lines += [
        "## Reproducing this", "", "```bash",
        "git clone <repo> && cd llm-infra-stack",
        "pip install -e packages/llmkit",
        "cd projects/15-public-teardown",
        "./teardown.py run --config configs/teardown.yaml",
        "./teardown.py verify --results results/",
        "./teardown.py publish --results results/ --out site/",
        "```", "",
        "Every run records its git SHA, workload fingerprint and the model the "
        "server actually reported serving. If a number here cannot be "
        "reproduced from that information, treat it as wrong and open an issue.",
        "", "See [METHODOLOGY.md](METHODOLOGY.md) for the measurement rules and "
        "their justification.", "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="measure every configuration")
    p.add_argument("--config", required=True)

    p = sub.add_parser("verify", help="publication gate")
    p.add_argument("--results", default="results")
    p.add_argument("--allow-caveats", action="store_true")

    p = sub.add_parser("publish", help="generate the public report")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="site")
    p.add_argument("--gpu", default="h100-sxm")
    p.add_argument("--n-gpus", type=int, default=1)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    return {"verify": cmd_verify, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
