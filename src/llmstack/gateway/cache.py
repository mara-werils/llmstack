"""Semantic response cache backed by Redis.

Caches LLM completions keyed by a hash of (model + messages).
Supports TTL-based expiration and cache hit/miss metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import redis.asyncio as aioredis

REDIS_URL = os.getenv("LLMSTACK_REDIS_URL", "")
CACHE_TTL = int(os.getenv("LLMSTACK_CACHE_TTL", "3600"))  # 1 hour default
CACHE_ENABLED = os.getenv("LLMSTACK_CACHE_ENABLED", "true").lower() == "true"

# Prefix to avoid key collisions
_KEY_PREFIX = "llmstack:cache:"
_STATS_KEY = "llmstack:cache:stats"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    avg_hit_latency_ms: float = 0.0
    _hit_latencies: list[float] = field(default_factory=list, repr=False)

    def record_hit(self, latency_ms: float) -> None:
        self.hits += 1
        self._hit_latencies.append(latency_ms)
        self.avg_hit_latency_ms = sum(self._hit_latencies) / len(self._hit_latencies)

    def record_miss(self) -> None:
        self.misses += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 2),
        }


class ResponseCache:
    """Redis-backed LLM response cache with semantic key hashing."""

    def __init__(self, redis_url: str = "", ttl: int = CACHE_TTL):
        self._url = redis_url or REDIS_URL
        self._ttl = ttl
        self._redis: aioredis.Redis | None = None
        self._stats = CacheStats()
        self._connected = False

    async def connect(self) -> None:
        """Lazily connect to Redis."""
        if self._connected or not self._url:
            return
        try:
            self._redis = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            await self._redis.ping()
            self._connected = True
        except Exception:
            self._redis = None
            self._connected = False

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._connected = False

    @staticmethod
    def _build_cache_key(model: str, messages: list[dict], temperature: float = 1.0) -> str:
        """Deterministic hash of the request for cache lookup.

        Only caches when temperature <= 0.1 (near-deterministic outputs).
        For higher temperatures, returns empty string (skip cache).
        """
        if temperature > 0.1:
            return ""

        # Normalize messages to a stable representation
        normalized = []
        for msg in messages:
            normalized.append({
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
            })

        payload = json.dumps({"model": model, "messages": normalized}, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"{_KEY_PREFIX}{digest}"

    async def get(self, model: str, messages: list[dict], temperature: float = 1.0) -> dict | None:
        """Look up a cached response. Returns None on miss."""
        if not self._connected or not CACHE_ENABLED:
            return None

        key = self._build_cache_key(model, messages, temperature)
        if not key:
            self._stats.record_miss()
            return None

        start = time.monotonic()
        try:
            raw = await self._redis.get(key)  # type: ignore[union-attr]
            if raw is None:
                self._stats.record_miss()
                return None

            latency_ms = (time.monotonic() - start) * 1000
            self._stats.record_hit(latency_ms)

            cached = json.loads(raw)
            # Mark response as cached
            cached["_cached"] = True
            cached["_cache_age_s"] = int(time.time() - cached.get("_cached_at", 0))
            return cached
        except Exception:
            self._stats.record_miss()
            return None

    async def put(self, model: str, messages: list[dict], response: dict,
                  temperature: float = 1.0) -> None:
        """Store a response in the cache."""
        if not self._connected or not CACHE_ENABLED:
            return

        key = self._build_cache_key(model, messages, temperature)
        if not key:
            return

        try:
            response["_cached_at"] = int(time.time())
            raw = json.dumps(response)
            await self._redis.set(key, raw, ex=self._ttl)  # type: ignore[union-attr]
        except Exception:
            pass  # Cache write failure is non-fatal

    async def invalidate(self, pattern: str = "*") -> int:
        """Delete cache entries matching a pattern. Returns count deleted."""
        if not self._connected:
            return 0
        try:
            keys = []
            async for key in self._redis.scan_iter(f"{_KEY_PREFIX}{pattern}"):  # type: ignore[union-attr]
                keys.append(key)
            if keys:
                return await self._redis.delete(*keys)  # type: ignore[union-attr]
            return 0
        except Exception:
            return 0

    @property
    def stats(self) -> CacheStats:
        return self._stats


# Module-level singleton
_cache: ResponseCache | None = None


async def get_cache() -> ResponseCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    if not _cache._connected:
        await _cache.connect()
    return _cache
