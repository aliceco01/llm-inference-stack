"""A faithful PagedAttention block allocator.

This is a working implementation of the mechanism, not a description of it:
fixed-size blocks, per-sequence block tables, content-addressed prefix sharing
with reference counts, copy-on-write on partial blocks, LRU eviction of
unreferenced cached blocks, and a watermark to avoid allocation deadlock.

A contiguous allocator is implemented alongside it (`ContiguousAllocator`) for
the comparison in project 09. The point of paging is not that it is clever, it
is that external fragmentation goes to zero: a request needs N blocks and any N
free blocks will do, so a fragmented pool still serves a large request that a
contiguous pool must reject despite having the bytes free.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def hash_block(prev_hash: str | None, token_ids: Sequence[int]) -> str:
    """Content address for a full block, chained to everything before it.

    Chaining matters: two requests share a cached block only if the entire
    preceding context is identical, otherwise position-dependent attention
    state would be reused across different prefixes and the output would be
    silently wrong.
    """
    h = hashlib.blake2b(digest_size=16)
    if prev_hash:
        h.update(prev_hash.encode())
    h.update(b"|")
    h.update(",".join(map(str, token_ids)).encode())
    return h.hexdigest()


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    content_hash: str | None = None   # set only when the block is full
    n_tokens: int = 0                 # tokens actually stored (<= block_size)
    last_used_step: int = 0

    @property
    def cached(self) -> bool:
        return self.content_hash is not None


class OutOfBlocks(Exception):
    pass


@dataclass
class AllocStats:
    allocated: int = 0
    freed: int = 0
    cache_hits: int = 0          # blocks satisfied from the prefix cache
    cache_misses: int = 0
    evictions: int = 0
    cow_copies: int = 0
    oom_events: int = 0


class PagedKVCache:
    """Block pool with content-addressed prefix sharing."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        *,
        enable_prefix_caching: bool = True,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.watermark_blocks = max(int(num_blocks * watermark), 1)

        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_ids: list[int] = list(range(num_blocks))
        # hash -> block_id, ordered by recency for LRU eviction
        self.cache: OrderedDict[str, int] = OrderedDict()
        self.stats = AllocStats()
        self.step = 0

    # ------------------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def utilization(self) -> float:
        return self.num_used / max(self.num_blocks, 1)

    @property
    def num_evictable(self) -> int:
        """Cached blocks nobody is currently using: reclaimable on demand."""
        return sum(1 for h, bid in self.cache.items() if self.blocks[bid].ref_count == 0)

    def can_allocate(self, n_blocks: int) -> bool:
        return self.num_free + self.num_evictable >= n_blocks + self.watermark_blocks

    # ------------------------------------------------------------------
    def _take_free_block(self) -> Block:
        """Pop a block, evicting the least recently used unreferenced cached
        block when the free list is empty."""
        if not self.free_ids:
            if not self._evict_one():
                self.stats.oom_events += 1
                raise OutOfBlocks("no free or evictable blocks")
        bid = self.free_ids.pop()
        blk = self.blocks[bid]
        blk.ref_count = 0
        blk.content_hash = None
        blk.n_tokens = 0
        self.stats.allocated += 1
        return blk

    def _evict_one(self) -> bool:
        for h, bid in list(self.cache.items()):
            if self.blocks[bid].ref_count == 0:
                del self.cache[h]
                self.blocks[bid].content_hash = None
                self.free_ids.append(bid)
                self.stats.evictions += 1
                return True
        return False

    def _release(self, bid: int) -> None:
        blk = self.blocks[bid]
        blk.ref_count -= 1
        if blk.ref_count <= 0:
            blk.ref_count = 0
            # A cached block stays resident and evictable: this is what makes
            # a later request with the same prefix a hit instead of a refill.
            if not (self.enable_prefix_caching and blk.cached):
                if blk.content_hash:
                    self.cache.pop(blk.content_hash, None)
                    blk.content_hash = None
                self.free_ids.append(bid)
                self.stats.freed += 1

    # ------------------------------------------------------------------
    def allocate_prompt(self, token_ids: Sequence[int]) -> tuple[list[int], int]:
        """Allocate the block table for a prompt.

        Returns (block_table, n_cached_tokens). Cached blocks are taken by
        reference, so a warm prefix costs zero new blocks and zero prefill
        compute for those tokens.
        """
        n = len(token_ids)
        n_full = n // self.block_size
        table: list[int] = []
        prev_hash: str | None = None
        cached_tokens = 0
        matching = self.enable_prefix_caching

        for i in range(n_full):
            chunk = token_ids[i * self.block_size:(i + 1) * self.block_size]
            bh = hash_block(prev_hash, chunk)
            prev_hash = bh
            if matching and bh in self.cache:
                bid = self.cache[bh]
                blk = self.blocks[bid]
                if blk.ref_count == 0 and bid in self.free_ids:
                    self.free_ids.remove(bid)
                blk.ref_count += 1
                blk.last_used_step = self.step
                self.cache.move_to_end(bh)
                table.append(bid)
                cached_tokens += self.block_size
                self.stats.cache_hits += 1
                continue
            # First miss ends the shared prefix: everything after it differs.
            matching = False
            self.stats.cache_misses += 1
            blk = self._take_free_block()
            blk.ref_count = 1
            blk.n_tokens = self.block_size
            blk.content_hash = bh
            blk.last_used_step = self.step
            if self.enable_prefix_caching:
                self.cache[bh] = blk.block_id
            table.append(blk.block_id)

        rem = n - n_full * self.block_size
        if rem:
            # Partial tail block is never cached: its content is still growing.
            blk = self._take_free_block()
            blk.ref_count = 1
            blk.n_tokens = rem
            table.append(blk.block_id)
        return table, cached_tokens

    def append_token(self, table: list[int], seq_len: int) -> bool:
        """Grow a sequence by one token. Returns True if a block was added.

        Decode allocates at most one block per step per sequence, and only when
        the current tail block fills. This is why decode memory pressure ramps
        in steps rather than smoothly.
        """
        offset = seq_len % self.block_size
        if offset == 0:
            blk = self._take_free_block()
            blk.ref_count = 1
            blk.n_tokens = 1
            table.append(blk.block_id)
            return True
        tail = self.blocks[table[-1]]
        if tail.ref_count > 1:
            # Copy-on-write: the tail is shared with another sequence and we
            # are about to write into it.
            new = self._take_free_block()
            new.ref_count = 1
            new.n_tokens = tail.n_tokens
            self._release(tail.block_id)
            table[-1] = new.block_id
            self.stats.cow_copies += 1
            tail = new
        tail.n_tokens = offset + 1
        return False

    def free(self, table: Iterable[int]) -> None:
        for bid in table:
            self._release(bid)

    # ------------------------------------------------------------------
    def fragmentation(self, seq_lens: Sequence[int]) -> dict[str, float]:
        """Internal fragmentation: bytes reserved but unused.

        Paging trades external fragmentation for a bounded amount of internal
        fragmentation: at most block_size-1 wasted token slots per sequence.
        With block_size=16 and any realistic sequence length that is under 1%,
        which is the whole argument for the design.
        """
        if not seq_lens:
            return {"internal_waste_tokens": 0, "internal_waste_pct": 0.0,
                    "external_waste_pct": 0.0}
        reserved = sum(
            ((l + self.block_size - 1) // self.block_size) * self.block_size
            for l in seq_lens
        )
        used = sum(seq_lens)
        return {
            "internal_waste_tokens": reserved - used,
            "internal_waste_pct": 100.0 * (reserved - used) / max(reserved, 1),
            # Paging cannot strand free memory: any free block serves any request.
            "external_waste_pct": 0.0,
        }

    def snapshot(self) -> dict[str, float | int]:
        return {
            "num_blocks": self.num_blocks,
            "used": self.num_used,
            "free": self.num_free,
            "utilization": round(self.utilization, 4),
            "cached_blocks": len(self.cache),
            "evictable": self.num_evictable,
            "hits": self.stats.cache_hits,
            "misses": self.stats.cache_misses,
            "evictions": self.stats.evictions,
            "cow": self.stats.cow_copies,
        }


class ContiguousAllocator:
    """Pre-paging baseline: each sequence gets one contiguous reservation.

    Included for the fragmentation comparison in project 09. Two failure modes
    show up immediately and neither exists under paging:

    * Over-reservation: the allocator cannot know the final length, so it must
      reserve max_model_len up front. A 100-token chat reserves 128k.
    * External fragmentation: free space exists but not contiguously, so a
      request is rejected while memory sits idle.
    """

    def __init__(self, capacity_tokens: int, reserve_len: int) -> None:
        self.capacity = capacity_tokens
        self.reserve_len = reserve_len
        self.slots: list[tuple[int, int, str | None]] = [(0, capacity_tokens, None)]
        self.rejections = 0

    def allocate(self, seq_id: str) -> int | None:
        need = self.reserve_len
        for i, (start, size, owner) in enumerate(self.slots):
            if owner is None and size >= need:
                self.slots[i] = (start, need, seq_id)
                if size > need:
                    self.slots.insert(i + 1, (start + need, size - need, None))
                return start
        self.rejections += 1
        return None

    def free(self, seq_id: str) -> None:
        for i, (start, size, owner) in enumerate(self.slots):
            if owner == seq_id:
                self.slots[i] = (start, size, None)
        self._coalesce()

    def _coalesce(self) -> None:
        merged: list[tuple[int, int, str | None]] = []
        for slot in sorted(self.slots):
            if merged and merged[-1][2] is None and slot[2] is None and \
                    merged[-1][0] + merged[-1][1] == slot[0]:
                s, sz, _ = merged[-1]
                merged[-1] = (s, sz + slot[1], None)
            else:
                merged.append(slot)
        self.slots = merged

    def stats(self, live_lens: dict[str, int]) -> dict[str, float]:
        used_slots = [(s, sz, o) for s, sz, o in self.slots if o is not None]
        reserved = sum(sz for _, sz, _ in used_slots)
        actually_used = sum(live_lens.get(o, 0) for _, _, o in used_slots)
        free_total = self.capacity - reserved
        largest_free = max((sz for _, sz, o in self.slots if o is None), default=0)
        return {
            "reserved_tokens": reserved,
            "used_tokens": actually_used,
            "internal_waste_pct": 100.0 * (reserved - actually_used) / max(reserved, 1),
            "free_tokens": free_total,
            "largest_free_run": largest_free,
            # Free bytes that cannot satisfy a full-size request.
            "external_waste_pct": 100.0 * (free_total - largest_free) / max(free_total, 1),
            "rejections": self.rejections,
        }
