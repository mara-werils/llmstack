"""Prompt prefix caching — reuse computation for shared prompt prefixes.

Detects common prefixes across requests (system prompts, context)
and caches their KV representations to speed up repeated requests.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock


@dataclass
class CachedPrefix:
    """A cached prompt prefix."""

    hash: str
    prefix_text: str
    token_count: int = 0
    hit_count: int = 0
    created_at: float = 0.0
    last_hit: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


class PromptPrefixCache:
    """LRU cache for prompt prefixes with automatic prefix detection.

    Caches the hash of common message prefixes (system prompt + context)
    so the gateway can hint to inference backends which parts to skip.
    """

    def __init__(self, max_entries: int = 500, min_prefix_length: int = 50):
        self._lock = Lock()
        self._cache: OrderedDict[str, CachedPrefix] = OrderedDict()
        self._max_entries = max_entries
        self._min_prefix_length = min_prefix_length
        self._total_hits = 0
        self._total_misses = 0

    @staticmethod
    def compute_prefix_hash(messages: list[dict], prefix_length: int = -1) -> str:
        """Compute a hash for the prefix of a message sequence.

        By default, hashes all messages except the last user message,
        which is typically the only varying part.
        """
        if not messages:
            return ""

        # The prefix is everything except the last user message
        if prefix_length < 0:
            prefix_msgs = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1 and msg.get("role") == "user":
                    break
                prefix_msgs.append(msg)
        else:
            prefix_msgs = messages[:prefix_length]

        if not prefix_msgs:
            return ""

        # Build stable hash
        parts = []
        for msg in prefix_msgs:
            parts.append(f"{msg.get('role', '')}:{msg.get('content', '')}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def lookup(self, messages: list[dict]) -> CachedPrefix | None:
        """Check if the prefix of these messages is cached."""
        prefix_hash = self.compute_prefix_hash(messages)
        if not prefix_hash:
            with self._lock:
                self._total_misses += 1
            return None

        with self._lock:
            entry = self._cache.get(prefix_hash)
            if entry is None:
                self._total_misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(prefix_hash)
            entry.hit_count += 1
            entry.last_hit = time.time()
            self._total_hits += 1
            return entry

    def store(self, messages: list[dict], token_count: int = 0) -> CachedPrefix | None:
        """Store a prefix in the cache."""
        prefix_hash = self.compute_prefix_hash(messages)
        if not prefix_hash:
            return None

        # Build prefix text
        prefix_msgs = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
        prefix_text = " | ".join(
            f"{m.get('role', '')}: {m.get('content', '')[:100]}"
            for m in prefix_msgs
        )

        if len(prefix_text) < self._min_prefix_length:
            return None

        with self._lock:
            if prefix_hash in self._cache:
                self._cache.move_to_end(prefix_hash)
                return self._cache[prefix_hash]

            entry = CachedPrefix(
                hash=prefix_hash,
                prefix_text=prefix_text[:500],
                token_count=token_count,
            )
            self._cache[prefix_hash] = entry

            # Evict LRU if over capacity
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

            return entry

    def invalidate(self, prefix_hash: str) -> bool:
        """Remove a specific prefix from the cache."""
        with self._lock:
            return self._cache.pop(prefix_hash, None) is not None

    def clear(self) -> int:
        """Clear all cached prefixes."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._total_hits + self._total_misses
            top_entries = sorted(
                self._cache.values(),
                key=lambda e: e.hit_count,
                reverse=True,
            )[:5]

            return {
                "total_entries": len(self._cache),
                "max_entries": self._max_entries,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": round(self._total_hits / total, 4) if total > 0 else 0.0,
                "top_prefixes": [
                    {
                        "hash": e.hash,
                        "hit_count": e.hit_count,
                        "prefix_preview": e.prefix_text[:100],
                    }
                    for e in top_entries
                ],
            }
