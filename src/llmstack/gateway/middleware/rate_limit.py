"""Token bucket rate limiter backed by Redis.

Supports per-key rate limiting with configurable burst.
Falls back to in-memory limiting if Redis is unavailable.
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from threading import Lock

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

REDIS_URL = os.getenv("LLMSTACK_REDIS_URL", "")
RATE_LIMIT = os.getenv("LLMSTACK_RATE_LIMIT", "100/min")

_TRUSTED_PROXIES: set[str] = {
    p.strip() for p in os.getenv("LLMSTACK_TRUSTED_PROXIES", "").split(",") if p.strip()
}

_RATE_PREFIX = "llmstack:ratelimit:"

# Lua script for atomic token bucket in Redis
# Returns: [allowed (0/1), remaining_tokens, retry_after_seconds]
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    last_refill = now
end

-- Refill tokens based on elapsed time
local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
local retry_after = 0

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    retry_after = math.ceil((requested - tokens) / refill_rate)
end

redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 10)

return {allowed, math.floor(tokens), retry_after}
"""


def _parse_rate_limit(spec: str) -> tuple[int, float]:
    """Parse rate limit spec like '100/min' into (capacity, refill_per_second).

    Supported formats: '100/min', '1000/hour', '10/sec'
    """
    match = re.match(r"(\d+)/(sec|min|hour|h|m|s)", spec.strip())
    if not match:
        return 100, 100 / 60  # default: 100/min

    count = int(match.group(1))
    unit = match.group(2)
    seconds = {"sec": 1, "s": 1, "min": 60, "m": 60, "hour": 3600, "h": 3600}[unit]

    return count, count / seconds


class _InMemoryBucket:
    """Fallback token bucket when Redis is unavailable."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = Lock()

    def try_acquire(self) -> tuple[bool, int, int]:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True, int(self.tokens), 0
            else:
                retry_after = int((1 - self.tokens) / self.refill_rate) + 1
                return False, 0, retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter with Redis backend and in-memory fallback."""

    SKIP_PATHS = {"/healthz", "/metrics", "/docs", "/openapi.json", "/"}

    def __init__(self, app, rate_limit: str = RATE_LIMIT):
        super().__init__(app)
        self.capacity, self.refill_rate = _parse_rate_limit(rate_limit)
        self._redis: aioredis.Redis | None = None
        self._lua_sha: str | None = None
        self._fallback_buckets: dict[str, _InMemoryBucket] = defaultdict(
            lambda: _InMemoryBucket(self.capacity, self.refill_rate)
        )
        self._redis_available = False

    async def _ensure_redis(self) -> bool:
        """Try to connect to Redis for distributed rate limiting."""
        if self._redis_available:
            return True
        if not REDIS_URL:
            return False
        try:
            self._redis = aioredis.from_url(
                REDIS_URL, decode_responses=True, socket_connect_timeout=2
            )
            await self._redis.ping()
            self._lua_sha = await self._redis.script_load(_TOKEN_BUCKET_LUA)
            self._redis_available = True
            return True
        except Exception:
            self._redis = None
            return False

    def _get_client_key(self, request: Request) -> str:
        """Extract rate limit key from request (API key or IP)."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            # Rate limit per API key
            return f"{_RATE_PREFIX}{auth[7:][:16]}"

        # Fallback to IP
        client = request.client
        ip = client.host if client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded and ip in _TRUSTED_PROXIES:
            ip = forwarded.split(",")[0].strip()
        return f"{_RATE_PREFIX}ip:{ip}"

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS or request.url.path.startswith("/ui"):
            return await call_next(request)

        client_key = self._get_client_key(request)

        # Try Redis first, fall back to in-memory
        if await self._ensure_redis():
            allowed, remaining, retry_after = await self._redis_check(client_key)
        else:
            allowed, remaining, retry_after = self._fallback_buckets[client_key].try_acquire()

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please retry later.",
                        "type": "rate_limit_error",
                        "retry_after": retry_after,
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.capacity),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.capacity)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    async def _redis_check(self, key: str) -> tuple[bool, int, int]:
        """Execute the Lua token bucket script atomically in Redis."""
        try:
            now = time.time()
            result = await self._redis.evalsha(  # type: ignore[union-attr]
                self._lua_sha,
                1,
                key,
                str(self.capacity),
                str(self.refill_rate),
                str(now),
                "1",
            )
            return bool(result[0]), int(result[1]), int(result[2])
        except Exception:
            # Redis failed mid-request, use fallback
            self._redis_available = False
            return self._fallback_buckets[key].try_acquire()
