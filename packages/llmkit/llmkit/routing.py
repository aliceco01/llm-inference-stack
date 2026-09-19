"""Replica selection strategies, shared by the prefix proxy (04) and the
gateway (13).

The central tension: prefix affinity wants to send every request with the same
system prompt to one replica so its KV blocks are reused, but load balancing
wants to spread requests evenly. Pure affinity produces hotspots (one popular
tenant saturates its replica while others idle); pure balancing destroys the
cache hit rate. The useful algorithm is bounded-load consistent hashing, which
takes affinity as a preference and abandons it when the preferred replica is
overloaded relative to the fleet.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Replica:
    """One backend, with the live state a router needs."""

    name: str
    base_url: str
    weight: float = 1.0
    healthy: bool = True

    inflight: int = 0
    queued_tokens: int = 0          # sum of prompt tokens in flight
    total_requests: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    ewma_ttft_ms: float = 0.0
    last_error_ts: float = 0.0
    open_until: float = 0.0         # circuit breaker

    # prefixes this replica is believed to hold, as a bounded LRU of keys
    prefix_keys: dict[str, float] = field(default_factory=dict)
    max_prefix_keys: int = 4096

    @property
    def available(self) -> bool:
        return self.healthy and time.time() >= self.open_until

    def note_prefix(self, key: str) -> None:
        self.prefix_keys[key] = time.time()
        if len(self.prefix_keys) > self.max_prefix_keys:
            oldest = sorted(self.prefix_keys.items(), key=lambda kv: kv[1])
            for k, _ in oldest[: len(oldest) // 4]:
                self.prefix_keys.pop(k, None)

    def has_prefix(self, key: str) -> bool:
        return key in self.prefix_keys

    def observe_ttft(self, ms: float, alpha: float = 0.2) -> None:
        self.ewma_ttft_ms = ms if not self.ewma_ttft_ms else (
            alpha * ms + (1 - alpha) * self.ewma_ttft_ms
        )

    def observe_error(self, *, breaker_threshold: int = 5, open_s: float = 10.0) -> None:
        self.total_errors += 1
        self.consecutive_errors += 1
        self.last_error_ts = time.time()
        if self.consecutive_errors >= breaker_threshold:
            # Trip the breaker. Continuing to send into a failing replica turns
            # one bad backend into fleet-wide latency, because every request
            # pays its timeout before failing over.
            self.open_until = time.time() + open_s

    def observe_success(self) -> None:
        self.consecutive_errors = 0


def prefix_key(system_prompt: str | None, prompt: str, *, n_chars: int = 2048) -> str:
    """Cache-affinity key.

    Keyed on the *leading* text, because that is what a prefix cache can
    actually share. Hashing the whole request would give every distinct request
    a distinct key and route as if by random.
    """
    head = (system_prompt or "")[:n_chars]
    if len(head) < n_chars:
        head += prompt[: n_chars - len(head)]
    return hashlib.blake2b(head.encode("utf-8", "ignore"), digest_size=12).hexdigest()


class Router(Protocol):
    name: str

    def pick(self, replicas: Sequence[Replica], key: str | None,
             est_tokens: int = 0) -> Replica | None: ...


def _available(replicas: Sequence[Replica]) -> list[Replica]:
    return [r for r in replicas if r.available]


class RoundRobinRouter:
    name = "round_robin"

    def __init__(self) -> None:
        self._c = itertools.count()

    def pick(self, replicas, key=None, est_tokens=0):
        av = _available(replicas)
        return av[next(self._c) % len(av)] if av else None


class RandomRouter:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def pick(self, replicas, key=None, est_tokens=0):
        av = _available(replicas)
        return self.rng.choice(av) if av else None


class LeastLoadedRouter:
    """Fewest in-flight requests, ties broken by queued prompt tokens.

    Queued tokens matter as a tiebreak because in-flight count treats a 32k
    prompt and a 32 token prompt as equal work, and they are not: prefill cost
    is linear in tokens.
    """

    name = "least_loaded"

    def pick(self, replicas, key=None, est_tokens=0):
        av = _available(replicas)
        if not av:
            return None
        return min(av, key=lambda r: (r.inflight / max(r.weight, 1e-6), r.queued_tokens))


class ConsistentHashRing:
    """Standard hash ring with virtual nodes.

    Virtual nodes exist so that adding or removing a replica moves ~1/N of keys
    instead of reshuffling everything. That matters here more than usual: every
    moved key is a prefix cache miss on the new replica and a wasted cached
    block on the old one.
    """

    def __init__(self, replicas: Iterable[Replica], vnodes: int = 160) -> None:
        self.vnodes = vnodes
        self._ring: list[tuple[int, str]] = []
        self.rebuild(replicas)

    def rebuild(self, replicas: Iterable[Replica]) -> None:
        ring: list[tuple[int, str]] = []
        for r in replicas:
            n = max(1, int(self.vnodes * r.weight))
            for i in range(n):
                h = hashlib.blake2b(f"{r.name}#{i}".encode(), digest_size=8).digest()
                ring.append((int.from_bytes(h, "big"), r.name))
        ring.sort()
        self._ring = ring
        self._keys = [k for k, _ in ring]

    def lookup(self, key: str, n: int = 1) -> list[str]:
        """The n distinct replica names clockwise from `key`."""
        if not self._ring:
            return []
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
        idx = bisect.bisect(self._keys, h) % len(self._ring)
        out: list[str] = []
        for i in range(len(self._ring)):
            name = self._ring[(idx + i) % len(self._ring)][1]
            if name not in out:
                out.append(name)
                if len(out) >= n:
                    break
        return out


class PrefixAffinityRouter:
    """Bounded-load consistent hashing on the prefix key.

    Behaviour:
      1. Hash the prefix to a preferred replica.
      2. Use it if its load is within `overload_factor` of the fleet mean.
      3. Otherwise walk the ring to the next replica, then fall back to
         least-loaded.

    Step 2 is what makes this usable. Pure affinity gives a popular system
    prompt its own permanently saturated replica while the rest idle, and the
    resulting queueing delay dwarfs the prefill saving the cache hit bought.
    The bound converts affinity from a constraint into a preference.
    """

    name = "prefix_affinity"

    def __init__(self, replicas: Sequence[Replica], *, overload_factor: float = 1.25,
                 vnodes: int = 160, probe: int = 3, honour_known_prefixes: bool = True) -> None:
        self.ring = ConsistentHashRing(replicas, vnodes=vnodes)
        self.overload_factor = overload_factor
        self.probe = probe
        self.honour_known_prefixes = honour_known_prefixes
        self.stats = {"affinity_hits": 0, "overload_deflections": 0,
                      "no_key": 0, "known_prefix_hits": 0}

    def rebuild(self, replicas: Sequence[Replica]) -> None:
        self.ring.rebuild(replicas)

    def pick(self, replicas, key=None, est_tokens=0):
        av = _available(replicas)
        if not av:
            return None
        if not key:
            self.stats["no_key"] += 1
            return LeastLoadedRouter().pick(av)

        by_name = {r.name: r for r in av}
        mean_load = sum(r.inflight for r in av) / len(av)
        ceiling = max(self.overload_factor * mean_load, 1.0)

        # A replica that demonstrably served this exact prefix recently beats
        # the ring, because the ring only guesses where the cache is.
        if self.honour_known_prefixes:
            holders = [r for r in av if r.has_prefix(key)]
            if holders:
                best = min(holders, key=lambda r: r.inflight)
                if best.inflight <= ceiling:
                    self.stats["known_prefix_hits"] += 1
                    return best

        for name in self.ring.lookup(key, n=self.probe):
            r = by_name.get(name)
            if r is None:
                continue
            if r.inflight <= ceiling:
                self.stats["affinity_hits"] += 1
                return r
        self.stats["overload_deflections"] += 1
        return LeastLoadedRouter().pick(av)


class SessionAffinityRouter:
    """Route by session id. Simple, and the right default for chat.

    A multi-turn conversation reuses the entire history as its prefix, so
    keeping a session on one replica gives near-perfect reuse without needing
    to reason about content at all.
    """

    name = "session_affinity"

    def __init__(self, replicas: Sequence[Replica], vnodes: int = 160,
                 overload_factor: float = 1.5) -> None:
        self.ring = ConsistentHashRing(replicas, vnodes=vnodes)
        self.overload_factor = overload_factor

    def rebuild(self, replicas: Sequence[Replica]) -> None:
        self.ring.rebuild(replicas)

    def pick(self, replicas, key=None, est_tokens=0):
        av = _available(replicas)
        if not av:
            return None
        if not key:
            return LeastLoadedRouter().pick(av)
        by_name = {r.name: r for r in av}
        mean_load = sum(r.inflight for r in av) / len(av)
        ceiling = max(self.overload_factor * mean_load, 1.0)
        for name in self.ring.lookup(key, n=3):
            r = by_name.get(name)
            if r and r.inflight <= ceiling:
                return r
        return LeastLoadedRouter().pick(av)


ROUTERS: dict[str, type] = {
    "round_robin": RoundRobinRouter,
    "random": RandomRouter,
    "least_loaded": LeastLoadedRouter,
    "prefix_affinity": PrefixAffinityRouter,
    "session_affinity": SessionAffinityRouter,
}


def make_router(name: str, replicas: Sequence[Replica], **kw):
    cls = ROUTERS.get(name)
    if cls is None:
        raise KeyError(f"unknown router {name!r}. Known: {', '.join(ROUTERS)}")
    if cls in (PrefixAffinityRouter, SessionAffinityRouter):
        return cls(replicas, **kw)
    return cls()
