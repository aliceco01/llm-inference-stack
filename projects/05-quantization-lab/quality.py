#!/usr/bin/env python3
"""Quality measurement for quantized serving.

Three measurements, in increasing order of what they actually tell you:

1. **Reference agreement.** Greedy-decode the same prompts against the fp16
   baseline and against the quantized model, and compare token by token.
   Reports exact-match rate and, more usefully, the *position* of first
   divergence: quantization damage compounds, so a model that diverges at
   token 3 is far worse than one that diverges at token 300 even if both score
   0% exact match.

2. **Logprob divergence.** Request top-k logprobs for the same forced
   continuation and compute KL(baseline || quantized) per token. This is the
   sensitive one: it detects degradation long before greedy output changes,
   because a token can stay argmax while its margin collapses.

3. **Task accuracy.** A small auto-scorable task set. Coarse, but it is the
   only one of the three that measures something a user would notice.

Why not perplexity: perplexity on wikitext is the standard quantization metric
and it is close to useless for serving decisions. It is dominated by common
tokens, insensitive to the long-generation degradation that actually breaks
agents, and routinely shows <1% change for schemes that visibly damage
multi-step reasoning. Measure agreement and task accuracy on traffic that looks
like yours.

    ./quality.py --baseline http://localhost:8000 --candidate http://localhost:8001 \
                 --model meta-llama/Llama-3.1-8B-Instruct --n 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# A small auto-scorable task set. Deliberately covers the failure modes that
# quantization hits hardest: multi-step arithmetic, long-range instruction
# following, and strict output formatting.
# ---------------------------------------------------------------------------
TASKS: list[dict[str, Any]] = [
    {"id": "arith-1", "kind": "exact",
     "prompt": "Compute 17 * 23 + 45. Reply with only the number.", "answer": "436"},
    {"id": "arith-2", "kind": "exact",
     "prompt": "Compute (144 / 12) * 7 - 19. Reply with only the number.", "answer": "65"},
    {"id": "arith-3", "kind": "exact",
     "prompt": "A train travels 60 km in 45 minutes. What is its speed in km/h? "
               "Reply with only the number.", "answer": "80"},
    {"id": "multistep-1", "kind": "exact",
     "prompt": "Start with 100. Subtract 37. Multiply by 2. Add 14. Divide by 6. "
               "Reply with only the final number.", "answer": "23"},
    {"id": "format-1", "kind": "json",
     "prompt": 'Return JSON with keys "city" and "country" for the Eiffel Tower. '
               "Reply with only JSON.",
     "answer": {"city": "Paris", "country": "France"}},
    {"id": "format-2", "kind": "json",
     "prompt": 'Return JSON with keys "a" and "b" where a=1 and b=2. Reply with only JSON.',
     "answer": {"a": 1, "b": 2}},
    {"id": "instruct-1", "kind": "contains_all",
     "prompt": "List exactly three primary colours, comma separated, lowercase, "
               "no other words.", "answer": ["red", "blue", "yellow"]},
    {"id": "instruct-2", "kind": "exact",
     "prompt": "Reply with exactly the word BANANA in uppercase and nothing else.",
     "answer": "BANANA"},
    {"id": "recall-1", "kind": "contains_all",
     "prompt": "What is the chemical symbol for gold? Reply with only the symbol.",
     "answer": ["Au"]},
    {"id": "negation-1", "kind": "exact",
     "prompt": "Which is larger, 9.11 or 9.9? Reply with only the number.",
     "answer": "9.9"},
]


@dataclass
class AgreementResult:
    n: int = 0
    exact_match: int = 0
    mean_first_divergence: float = 0.0
    median_first_divergence: float = 0.0
    divergence_positions: list[int] = field(default_factory=list)
    mean_prefix_agreement: float = 0.0

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / self.n if self.n else float("nan")


@dataclass
class TaskResult:
    total: int = 0
    correct: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else float("nan")


async def generate(client: httpx.AsyncClient, base_url: str, model: str,
                   prompt: str, *, max_tokens: int = 128,
                   logprobs: int = 0, api_key: str | None = None) -> dict[str, Any]:
    """Greedy, deterministic generation. seed is set where the backend honours it."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = logprobs
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = await client.post(base_url.rstrip("/") + "/v1/chat/completions",
                          json=body, headers=headers, timeout=300.0)
    r.raise_for_status()
    return r.json()


def _text_of(resp: dict[str, Any]) -> str:
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return (msg.get("content") or ch.get("text") or "").strip()


def _token_list(resp: dict[str, Any]) -> list[str]:
    """Per-token strings, when the backend returns logprobs."""
    ch = (resp.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    content = lp.get("content") or []
    return [c.get("token", "") for c in content]


def _topk_dist(resp: dict[str, Any], idx: int) -> dict[str, float]:
    """Top-k distribution at position idx, as probabilities."""
    ch = (resp.get("choices") or [{}])[0]
    content = (ch.get("logprobs") or {}).get("content") or []
    if idx >= len(content):
        return {}
    tops = content[idx].get("top_logprobs") or []
    return {t["token"]: math.exp(t["logprob"]) for t in tops if "token" in t}


def kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    """KL(p || q) over the union of the two top-k supports.

    Truncated top-k makes this an approximation: mass outside both supports is
    unaccounted for. It is still the right relative signal for comparing
    schemes against one baseline, which is what it is used for.
    """
    if not p:
        return float("nan")
    keys = set(p) | set(q)
    zp = sum(p.get(k, 0.0) for k in keys) or 1.0
    zq = sum(q.get(k, 0.0) for k in keys) or 1.0
    out = 0.0
    for k in keys:
        pi = p.get(k, 0.0) / zp
        qi = q.get(k, 0.0) / zq
        if pi > 0:
            out += pi * math.log(pi / max(qi, eps))
    return out


async def measure_agreement(baseline: str, candidate: str, model: str,
                            prompts: list[str], *, max_tokens: int,
                            api_key: str | None) -> AgreementResult:
    res = AgreementResult()
    async with httpx.AsyncClient() as c:
        for prompt in prompts:
            try:
                a, b = await asyncio.gather(
                    generate(c, baseline, model, prompt, max_tokens=max_tokens, api_key=api_key),
                    generate(c, candidate, model, prompt, max_tokens=max_tokens, api_key=api_key),
                )
            except Exception as e:
                print(f"  generation failed: {type(e).__name__}: {e}")
                continue
            ta, tb = _token_list(a), _token_list(b)
            if not ta or not tb:
                # No logprobs: fall back to comparing raw text by whitespace.
                ta, tb = _text_of(a).split(), _text_of(b).split()
            res.n += 1
            if ta == tb:
                res.exact_match += 1
                res.divergence_positions.append(len(ta))
                continue
            div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                       min(len(ta), len(tb)))
            res.divergence_positions.append(div)
    if res.divergence_positions:
        res.mean_first_divergence = statistics.mean(res.divergence_positions)
        res.median_first_divergence = statistics.median(res.divergence_positions)
        res.mean_prefix_agreement = statistics.mean(
            d / max(max_tokens, 1) for d in res.divergence_positions)
    return res


async def measure_kl(baseline: str, candidate: str, model: str,
                     prompts: list[str], *, max_tokens: int, top_k: int,
                     api_key: str | None) -> dict[str, float]:
    kls: list[float] = []
    async with httpx.AsyncClient() as c:
        for prompt in prompts:
            try:
                a, b = await asyncio.gather(
                    generate(c, baseline, model, prompt, max_tokens=max_tokens,
                             logprobs=top_k, api_key=api_key),
                    generate(c, candidate, model, prompt, max_tokens=max_tokens,
                             logprobs=top_k, api_key=api_key),
                )
            except Exception as e:
                print(f"  logprob request failed ({type(e).__name__}); "
                      "backend may not support top_logprobs")
                return {}
            n = min(len(_token_list(a)), len(_token_list(b)))
            for i in range(n):
                v = kl_divergence(_topk_dist(a, i), _topk_dist(b, i))
                if not math.isnan(v):
                    kls.append(v)
    if not kls:
        return {}
    kls.sort()
    return {
        "n_tokens": len(kls),
        "mean_kl": statistics.mean(kls),
        "median_kl": statistics.median(kls),
        "p95_kl": kls[max(0, math.ceil(0.95 * len(kls)) - 1)],
        "max_kl": kls[-1],
    }


def score_task(task: dict[str, Any], output: str) -> bool:
    kind = task["kind"]
    if kind == "exact":
        cleaned = output.strip().strip(".").strip()
        return cleaned == task["answer"] or cleaned.split()[-1:] == [task["answer"]]
    if kind == "contains_all":
        low = output.lower()
        return all(a.lower() in low for a in task["answer"])
    if kind == "json":
        txt = output.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            got = json.loads(txt)
        except Exception:
            return False
        return all(str(got.get(k)).lower() == str(v).lower()
                   for k, v in task["answer"].items())
    return False


async def measure_tasks(url: str, model: str, *, api_key: str | None,
                        repeats: int = 1) -> TaskResult:
    res = TaskResult()
    async with httpx.AsyncClient() as c:
        for _ in range(repeats):
            for t in TASKS:
                try:
                    r = await generate(c, url, model, t["prompt"], max_tokens=64,
                                       api_key=api_key)
                except Exception as e:
                    res.total += 1
                    res.failures.append(f"{t['id']}: request failed ({type(e).__name__})")
                    continue
                out = _text_of(r)
                res.total += 1
                if score_task(t, out):
                    res.correct += 1
                else:
                    res.failures.append(f"{t['id']}: got {out[:60]!r}")
    return res


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="fp16/bf16 reference endpoint")
    ap.add_argument("--candidate", required=True, help="quantized endpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=32, help="agreement prompts")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--skip-kl", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from llmkit.workload import make_text
    prompts = [
        f"Summarise the following in one sentence:\n\n{make_text(180, seed=i)}"
        for i in range(args.n)
    ]

    print("measuring reference agreement ...")
    agree = await measure_agreement(args.baseline, args.candidate, args.model,
                                    prompts, max_tokens=args.max_tokens,
                                    api_key=args.api_key)
    print(f"  n                      {agree.n}")
    print(f"  exact match            {agree.exact_match_rate*100:.1f}%")
    print(f"  first divergence       mean {agree.mean_first_divergence:.1f} "
          f"/ median {agree.median_first_divergence:.1f} tokens")
    print(f"  mean prefix agreement  {agree.mean_prefix_agreement*100:.1f}% of output")

    kl: dict[str, float] = {}
    if not args.skip_kl:
        print("\nmeasuring logprob divergence ...")
        kl = await measure_kl(args.baseline, args.candidate, args.model,
                              prompts[: max(4, args.n // 4)],
                              max_tokens=min(args.max_tokens, 64),
                              top_k=args.top_k, api_key=args.api_key)
        if kl:
            print(f"  tokens compared        {kl['n_tokens']}")
            print(f"  KL mean / median       {kl['mean_kl']:.5f} / {kl['median_kl']:.5f}")
            print(f"  KL p95 / max           {kl['p95_kl']:.5f} / {kl['max_kl']:.5f}")
        else:
            print("  skipped (backend did not return top_logprobs)")

    print("\nmeasuring task accuracy ...")
    base_tasks = await measure_tasks(args.baseline, args.model, api_key=args.api_key)
    cand_tasks = await measure_tasks(args.candidate, args.model, api_key=args.api_key)
    print(f"  baseline   {base_tasks.correct}/{base_tasks.total} "
          f"= {base_tasks.accuracy*100:.1f}%")
    print(f"  candidate  {cand_tasks.correct}/{cand_tasks.total} "
          f"= {cand_tasks.accuracy*100:.1f}%")
    delta = cand_tasks.accuracy - base_tasks.accuracy
    print(f"  delta      {delta*100:+.1f} points")
    if cand_tasks.failures:
        print("  candidate failures:")
        for f in cand_tasks.failures[:8]:
            print(f"    {f}")

    print("\nverdict:")
    if agree.exact_match_rate >= 0.95 and delta >= -0.02:
        print("  near-lossless on this task set. Note the set is small; confirm on "
              "traffic that resembles production before shipping.")
    elif agree.mean_prefix_agreement >= 0.8 and delta >= -0.05:
        print("  measurable drift, task accuracy broadly held. Acceptable for many "
              "workloads; not for agents that chain many steps.")
    else:
        print("  material quality loss. Do not ship on the strength of a latency win.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "baseline": args.baseline, "candidate": args.candidate,
            "model": args.model,
            "agreement": {
                "n": agree.n, "exact_match_rate": agree.exact_match_rate,
                "mean_first_divergence": agree.mean_first_divergence,
                "median_first_divergence": agree.median_first_divergence,
                "mean_prefix_agreement": agree.mean_prefix_agreement,
            },
            "kl": kl,
            "tasks": {"baseline_accuracy": base_tasks.accuracy,
                      "candidate_accuracy": cand_tasks.accuracy,
                      "delta": delta},
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
