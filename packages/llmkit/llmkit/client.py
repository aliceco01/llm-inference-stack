"""Async OpenAI-compatible streaming client instrumented for latency work.

This is the single measurement surface for the whole repo. vLLM, SGLang,
TGI, the local simulator, and hosted providers all speak this protocol, so one
correct client covers every backend and the numbers stay comparable.

Correctness details that ordinary clients get wrong:

* The first SSE frame from an OpenAI-compatible server is usually a role-only
  delta with empty content. Treating it as the first token under-reports TTFT
  by one scheduler step, which at high concurrency is tens of milliseconds.
  We record both and report the content-based one.
* A chunk can carry more than one token. We use server `usage` when present
  and otherwise count the delta text, rather than assuming one frame is one
  token.
* The timestamp is taken the instant the frame is parsed off the socket, before
  any bookkeeping, so client-side work is not folded into ITL.
* Output length must be *controlled* for a benchmark to be comparable. We send
  `ignore_eos`/`min_tokens` when the backend supports them so every request
  produces exactly the requested number of decode steps.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from .tokenizer import Counter, get_counter
from .types import FinishReason, RequestRecord, TokenEvent, now_ns


@dataclass
class EndpointConfig:
    """How to reach one backend."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = "mock-model"
    api_key: str | None = None
    api: str = "chat"                    # "chat" | "completions"
    timeout_s: float = 600.0
    connect_timeout_s: float = 10.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Backends differ in how they force a fixed output length.
    supports_ignore_eos: bool = True
    supports_min_tokens: bool = True
    name: str = "default"

    def url(self) -> str:
        path = "/v1/chat/completions" if self.api == "chat" else "/v1/completions"
        return self.base_url.rstrip("/") + path

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h


@dataclass
class Request:
    """One unit of work to send."""

    prompt: str
    max_tokens: int = 128
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    system: str | None = None
    temperature: float = 0.0
    session_id: str | None = None
    tenant: str | None = None
    prompt_tokens_hint: int = 0     # known length for synthetic workloads
    fixed_output: bool = True       # force exactly max_tokens decode steps
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)


def build_payload(req: Request, ep: EndpointConfig) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": ep.model,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
        # Ask for usage on the final frame. Servers that do not support it
        # ignore the field, and we fall back to counting text.
        "stream_options": {"include_usage": True},
    }
    if ep.api == "chat":
        msgs = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        msgs.append({"role": "user", "content": req.prompt})
        body["messages"] = msgs
    else:
        full = (req.system + "\n\n" + req.prompt) if req.system else req.prompt
        body["prompt"] = full

    if req.fixed_output:
        # Controlled decode length. Without this, a model that emits EOS early
        # turns a "512 output token" benchmark into a 40 token benchmark and
        # the throughput number becomes meaningless.
        if ep.supports_ignore_eos:
            body["ignore_eos"] = True
        if ep.supports_min_tokens:
            body["min_tokens"] = req.max_tokens

    body.update(ep.extra_body)
    body.update(req.extra_body)
    return body


def _extract_delta(obj: dict[str, Any], api: str) -> tuple[str, str | None]:
    """Return (text, finish_reason) from one streamed chunk."""
    choices = obj.get("choices") or []
    if not choices:
        return "", None
    c0 = choices[0]
    fr = c0.get("finish_reason")
    if api == "chat":
        d = c0.get("delta") or {}
        return d.get("content") or "", fr
    return c0.get("text") or "", fr


class StreamingClient:
    """Reusable async client. One instance per process; share the pool."""

    def __init__(
        self,
        ep: EndpointConfig,
        *,
        counter: Counter | None = None,
        max_connections: int = 2048,
    ) -> None:
        self.ep = ep
        self.counter = counter or get_counter(ep.model)
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(ep.timeout_s, connect=ep.connect_timeout_s),
            http2=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> StreamingClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def send(self, req: Request, *, submitted_ns: int | None = None) -> RequestRecord:
        ep = self.ep
        rec = RequestRecord(
            request_id=req.request_id,
            model=ep.model,
            tenant=req.tenant,
            session_id=req.session_id,
        )
        rec.t_submit_ns = submitted_ns if submitted_ns is not None else now_ns()
        payload = build_payload(req, ep)
        headers = {**ep.headers(), **req.extra_headers,
                   "x-request-id": req.request_id}

        text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish: str | None = None
        n_content_chunks = 0

        try:
            rec.t_send_ns = now_ns()
            async with self._client.stream(
                "POST", ep.url(), json=payload, headers=headers
            ) as resp:
                rec.status_code = resp.status_code
                rec.replica = resp.headers.get("x-replica") or resp.headers.get("x-backend")
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    rec.error = f"http_{resp.status_code}: {body}"
                    rec.finish_reason = FinishReason.ERROR
                    rec.t_done_ns = now_ns()
                    return rec

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    t = now_ns()  # stamp first, parse second
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if rec.t_first_chunk_ns < 0:
                        rec.t_first_chunk_ns = t
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    delta, fr = _extract_delta(obj, ep.api)
                    if fr:
                        finish = fr
                    if not delta:
                        continue
                    n_content_chunks += 1
                    if rec.t_first_token_ns < 0:
                        rec.t_first_token_ns = t
                    rec.t_last_token_ns = t
                    text_parts.append(delta)
                    rec.token_events.append(TokenEvent(t_ns=t, content_tokens=1, text=delta))

            rec.t_done_ns = now_ns()

        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            rec.error = f"timeout: {type(e).__name__}"
            rec.finish_reason = FinishReason.TIMEOUT
            rec.t_done_ns = now_ns()
            return rec
        except Exception as e:  # connection reset, DNS, protocol error
            rec.error = f"{type(e).__name__}: {e}"[:200]
            rec.finish_reason = FinishReason.ERROR
            rec.t_done_ns = now_ns()
            return rec

        full = "".join(text_parts)
        # Prefer server-reported usage; it is exact and it is what gets billed.
        if usage:
            rec.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            rec.output_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            rec.cached_prompt_tokens = int(details.get("cached_tokens") or 0)
            rec.extra["token_source"] = "server_usage"
        else:
            rec.prompt_tokens = req.prompt_tokens_hint or self.counter.count(
                (req.system or "") + req.prompt
            )
            rec.output_tokens = self.counter.count(full)
            rec.extra["token_source"] = getattr(self.counter, "name", "heuristic")

        # Reconcile: if the server says N tokens but we saw M content frames,
        # the frames were multi-token. Redistribute so ITL stays honest.
        if rec.output_tokens and n_content_chunks and rec.output_tokens != n_content_chunks:
            self._rebalance_token_events(rec, rec.output_tokens, n_content_chunks)

        rec.finish_reason = (
            FinishReason.LENGTH if finish == "length" else FinishReason.STOP
        )
        rec.extra["n_stream_chunks"] = n_content_chunks
        return rec

    @staticmethod
    def _rebalance_token_events(rec: RequestRecord, total: int, chunks: int) -> None:
        """Spread `total` tokens across `chunks` observed frames.

        Servers under load coalesce several decoded tokens into one SSE frame.
        Leaving each frame as one token makes ITL look uniform and wrong; this
        assigns the real token count so the amortisation in
        RequestRecord.itls_ms produces the true per-token spacing.
        """
        if chunks <= 0 or total <= 0:
            return
        base, rem = divmod(total, chunks)
        for i, ev in enumerate(rec.token_events):
            ev.content_tokens = base + (1 if i < rem else 0)
        rec.extra["tokens_per_chunk_mean"] = total / chunks


async def probe(ep: EndpointConfig, *, timeout_s: float = 5.0) -> dict[str, Any]:
    """Liveness/identity probe used before every benchmark run.

    A load test against a backend serving a different model or a different
    max-model-len than you believe is the most expensive kind of wrong number,
    so we record what actually answered.
    """
    out: dict[str, Any] = {"base_url": ep.base_url, "reachable": False}
    async with httpx.AsyncClient(timeout=timeout_s) as c:
        for path in ("/v1/models", "/health", "/"):
            try:
                r = await c.get(ep.base_url.rstrip("/") + path, headers=ep.headers())
                if r.status_code < 400:
                    out["reachable"] = True
                    out["probe_path"] = path
                    try:
                        out["body"] = r.json()
                    except Exception:
                        out["body"] = r.text[:200]
                    break
            except Exception as e:
                out["last_error"] = f"{type(e).__name__}: {e}"[:120]
    return out
