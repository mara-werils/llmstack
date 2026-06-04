"""Request deduplication for idempotent API calls.

Prevents duplicate processing of identical requests within a time window
using idempotency keys. Useful for retry scenarios where clients may
resend the same request.
"""

from __future__ import annotations

import hashlib
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached response for deduplication."""

    idempotency_key: str
    status_code: int
    body: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class DedupConfig:
    """Configuration for request deduplication."""

    # TTL for cached responses (seconds)
    ttl: float = 300.0  # 5 minutes

    # Maximum cached responses
    max_entries: int = 10000

    # Whether to auto-generate keys from request content
    auto_key: bool = True


class RequestDeduplicator:
    """Deduplicates API requests using idempotency keys.

    Caches responses for identical requests within a time window,
    returning the cached response for duplicate requests.
    """

    def __init__(self, config: DedupConfig | None = None):
        self.config = config or DedupConfig()
        self._cache: dict[str, CachedResponse] = {}
        self._lock = threading.Lock()

    def get_cached(self, key: str) -> CachedResponse | None:
        """Check if a response is cached for this idempotency key."""
        with self._lock:
            cached = self._cache.get(key)
            if cached and (time.time() - cached.created_at) < self.config.ttl:
                return cached
            if cached:
                del self._cache[key]
            return None

    def cache_response(
        self,
        key: str,
        status_code: int,
        body: dict[str, Any],
    ) -> None:
        """Cache a response for future deduplication."""
        with self._lock:
            self._cache[key] = CachedResponse(
                idempotency_key=key,
                status_code=status_code,
                body=body,
            )
            self._evict_if_needed()

    def generate_key(self, method: str, path: str, body: str = "") -> str:
        """Generate an idempotency key from request content."""
        content = f"{method}:{path}:{body}"
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        with self._lock:
            now = time.time()
            active = sum(1 for c in self._cache.values() if (now - c.created_at) < self.config.ttl)
            return {
                "total_cached": len(self._cache),
                "active_entries": active,
                "max_entries": self.config.max_entries,
                "ttl_seconds": self.config.ttl,
            }

    def clear(self) -> int:
        """Clear all cached responses. Returns count removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over max."""
        while len(self._cache) > self.config.max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
