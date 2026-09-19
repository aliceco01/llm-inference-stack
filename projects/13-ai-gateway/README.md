# 13 - AI Gateway with Fallbacks and Rate Limits

> Provider outages are guaranteed. User-facing errors are optional.

An OpenAI-compatible gateway that routes across self-hosted replicas and API
providers, enforces per-tenant limits, holds TTFT SLOs, and degrades through a
configured chain instead of returning 503s.

```bash
./gateway.py --config config.yaml --print-chain    # show the fallback order
./gateway.py --config config.yaml --port 9000
curl -s localhost:9000/stats | jq
```

## The degradation chain ends in an answer, not an error

```
primary         vllm-a, vllm-b                (self-hosted)
secondary       provider-same-model           (paid API, same model)
cheaper_model   provider-small-model          (worse answers, still answers)
cached          responses up to 3600s old
```

A gateway whose last resort is a 503 has not degraded, it has failed. Serving a
1B model's answer during an outage is worse than the 8B answer and much better
than nothing, and deciding that **in advance** is what makes an outage
survivable. Responses carry `x-gateway-tier` and `x-gateway-degraded` so callers
can see they are on a fallback path and react if they care.

Within a tier, the least loaded backend that is **meeting its TTFT SLO** wins.
A backend that is up but slow is not a healthy backend.

## Retry budgets, not retry counts

This is the part most gateways get wrong.

A per-request retry limit means that when a backend degrades, *every* request
retries, and offered load multiplies by the retry count at exactly the moment
capacity is insufficient. That converts a partial outage into a total one.

The budget caps retries as a **fraction of total traffic** over a sliding
window (default 10%, Google SRE's pattern). Retries stay available for isolated
failures and cannot amplify a systemic one. `/stats` exposes the live retry
ratio and how many retries were denied.

Backoff uses **full jitter** (uniform in `[0, delay]`) rather than fixed
exponential, because synchronised clients retrying at the same computed instant
recreate the thundering herd the backoff was meant to prevent.

## Rate limits need two dimensions

LLM traffic is badly described by requests per minute: one request can be 200
tokens or 200,000. Limiting RPM alone lets a single tenant with long prompts
saturate the fleet while staying inside quota.

Every tenant has a token bucket on **requests** and one on **tokens**, plus a
concurrency cap and an optional hourly dollar budget.

Two implementation details that matter:

- **Rejections refund the other bucket.** A token-limit rejection refunds the
  request token it already consumed, or a tenant's effective RPM would silently
  depend on its prompt sizes.
- **Estimates are settled after the response.** Completion length is unknown at
  admission, so the admission estimate is always wrong. `settle()` reconciles
  it, otherwise tenants with long outputs systematically underpay.

The example config shows the useful shape: `interactive-app` gets high RPM and
concurrency, `batch-jobs` gets 20x the token throughput but `max_concurrent: 8`
so it cannot crowd interactive requests out of the engine's running batch.

## Hedging, and why the loser must be cancelled

Hedging sends a duplicate request when the first has not produced a token by a
deadline. It is one of the most effective tail-latency tools available, with two
conditions:

1. **Set `delay_ms` near the primary's p95 TTFT, not its mean.** At p95 the
   extra load is ~5% and it targets exactly the requests that were going to be
   slow. At the mean you double fleet load to fix nothing.
2. **Cancel the loser.** `hedged_call()` cancels and awaits the pending task. An
   uncancelled hedge is a permanent load increase proportional to the hedge
   rate, which raises p95, which triggers more hedging.

Disabled by default in the config, because it is only safe for work without
side effects.

## Retry safety on streaming responses

`forward()` opens the upstream stream and checks its status **before** returning
a `StreamingResponse`. A connection failure or 5xx can therefore be retried on
another backend. Once tokens have been forwarded to the client, retrying would
duplicate output, so it does not. This is the same discipline as the project 04
proxy and it is the difference between a safe retry and a corrupted response.

## Circuit breakers

Five consecutive failures opens a backend for 15 seconds. Without this, one
wedged backend makes every request pay its full timeout before failing over,
turning one bad provider into fleet-wide latency.

Health probes and request outcomes are weighted differently, on purpose. A
replica can answer `/health` in two milliseconds while its engine is wedged, so
**request outcomes are the primary signal**: five consecutive request failures
open the breaker. A failed probe is weaker evidence and needs three consecutive
failures to open it.

But a failed probe must count for *something*. An earlier version of this
gateway only ever marked backends healthy and never unhealthy, so a replica that
was down reported `available: true` in `/health` until a user request found it
and paid a connect timeout first. Both counters are exposed separately in
`/stats` as `consecutive_errors` and `probe_failures`.

Paid API backends are **not** polled at all. A synthetic health request to a
metered endpoint costs money and tells you less than real outcomes do, so their
breaker is driven entirely by observed results.

`POST /admin/backend/{name}/disable` opens a breaker manually, which is what
project 14's chaos drills use and what you want during a real incident.

## Observability

`/stats` gives a human-readable view: per-backend inflight, errors, breaker
state, EWMA TTFT and SLO status; per-tier request counts; retry budget state;
per-tenant spend. `/metrics` exposes the same in Prometheus format, including
`gateway_degraded_total`, which is the single most important alert on this
service: it means users are being served by a fallback and nobody has noticed.
