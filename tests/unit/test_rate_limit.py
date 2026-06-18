"""Tests for token bucket rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.testclient import TestClient

from llmstack.gateway.middleware.rate_limit import (
    RateLimitMiddleware,
    _InMemoryBucket,
    _parse_rate_limit,
)


class TestRateLimitParsing:
    """Test rate limit spec parsing."""

    def test_per_minute(self):
        capacity, refill = _parse_rate_limit("100/min")
        assert capacity == 100
        assert refill == pytest.approx(100 / 60)

    def test_per_second(self):
        capacity, refill = _parse_rate_limit("10/sec")
        assert capacity == 10
        assert refill == 10.0

    def test_per_hour(self):
        capacity, refill = _parse_rate_limit("3600/hour")
        assert capacity == 3600
        assert refill == pytest.approx(1.0)

    def test_short_units(self):
        cap_m, rate_m = _parse_rate_limit("60/m")
        cap_s, rate_s = _parse_rate_limit("1/s")
        cap_h, rate_h = _parse_rate_limit("3600/h")
        assert cap_m == 60
        assert cap_s == 1
        assert cap_h == 3600

    def test_invalid_returns_default(self):
        capacity, refill = _parse_rate_limit("invalid")
        assert capacity == 100
        assert refill == pytest.approx(100 / 60)


class TestInMemoryBucket:
    """Test the in-memory fallback token bucket."""

    def test_allows_within_capacity(self):
        bucket = _InMemoryBucket(capacity=5, refill_rate=1.0)
        for _ in range(5):
            allowed, remaining, retry = bucket.try_acquire()
            assert allowed is True

    def test_rejects_over_capacity(self):
        bucket = _InMemoryBucket(capacity=2, refill_rate=0.0001)
        bucket.try_acquire()
        bucket.try_acquire()
        allowed, remaining, retry = bucket.try_acquire()
        assert allowed is False
        assert retry > 0

    def test_remaining_decrements(self):
        bucket = _InMemoryBucket(capacity=3, refill_rate=0.0)
        _, r1, _ = bucket.try_acquire()
        _, r2, _ = bucket.try_acquire()
        assert r1 > r2

    def test_refills_over_time(self):
        """Bucket should refill tokens based on elapsed time."""
        import time

        bucket = _InMemoryBucket(capacity=1, refill_rate=1000.0)  # fast refill
        bucket.try_acquire()
        time.sleep(0.01)  # allow 10ms to elapse for refill
        allowed, _, _ = bucket.try_acquire()
        assert allowed is True


def _app(rate_limit="100/min"):
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, rate_limit=rate_limit)

    @app.get("/v1/models")
    async def models():
        return {"ok": True}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    return app


class TestDispatch:
    def test_skip_paths_bypass_rate_limiting(self):
        client = TestClient(_app("1/sec"))
        for _ in range(5):
            assert client.get("/healthz").status_code == 200

    def test_allowed_request_sets_headers(self):
        client = TestClient(_app("100/min"))
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.headers["X-RateLimit-Limit"] == "100"

    def test_uses_redis_check_when_redis_available(self):
        with (
            patch(
                "llmstack.gateway.middleware.rate_limit.RateLimitMiddleware._ensure_redis",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "llmstack.gateway.middleware.rate_limit.RateLimitMiddleware._redis_check",
                new=AsyncMock(return_value=(True, 42, 0)),
            ),
        ):
            client = TestClient(_app("100/min"))
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.headers["X-RateLimit-Remaining"] == "42"

    def test_blocked_request_returns_429(self):
        client = TestClient(_app("1/sec"))
        first = client.get("/v1/models")
        second = client.get("/v1/models")
        assert first.status_code == 200
        assert second.status_code == 429
        assert "Retry-After" in second.headers
        assert second.json()["error"]["type"] == "rate_limit_error"


class TestGetClientKey:
    def test_uses_bearer_token(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        scope = {
            "type": "http",
            "headers": [(b"authorization", b"Bearer my-secret-token-1234567890")],
            "client": ("1.2.3.4", 1),
        }
        key = mw._get_client_key(Request(scope))
        assert key == "llmstack:ratelimit:" + "my-secret-token-1234567890"[:16]

    def test_falls_back_to_ip_when_no_bearer(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        scope = {"type": "http", "headers": [], "client": ("9.8.7.6", 1)}
        key = mw._get_client_key(Request(scope))
        assert key == "llmstack:ratelimit:ip:9.8.7.6"

    def test_ignores_forwarded_for_when_proxy_not_trusted(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        scope = {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"1.2.3.4")],
            "client": ("9.8.7.6", 1),
        }
        key = mw._get_client_key(Request(scope))
        assert key == "llmstack:ratelimit:ip:9.8.7.6"

    def test_uses_forwarded_for_when_proxy_trusted(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        scope = {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")],
            "client": ("9.8.7.6", 1),
        }
        with patch("llmstack.gateway.middleware.rate_limit._TRUSTED_PROXIES", {"9.8.7.6"}):
            key = mw._get_client_key(Request(scope))
        assert key == "llmstack:ratelimit:ip:1.2.3.4"


class TestEnsureRedis:
    @pytest.mark.asyncio
    async def test_no_redis_url_returns_false(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        with patch("llmstack.gateway.middleware.rate_limit.REDIS_URL", ""):
            assert await mw._ensure_redis() is False

    @pytest.mark.asyncio
    async def test_connects_and_caches(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        fake_redis = AsyncMock()
        fake_redis.script_load = AsyncMock(return_value="sha123")
        with (
            patch("llmstack.gateway.middleware.rate_limit.REDIS_URL", "redis://localhost:6379"),
            patch(
                "llmstack.gateway.middleware.rate_limit.aioredis.from_url",
                return_value=fake_redis,
            ) as mock_from_url,
        ):
            first = await mw._ensure_redis()
            second = await mw._ensure_redis()
        assert first is True
        assert second is True
        mock_from_url.assert_called_once()  # second call hit the cache

    @pytest.mark.asyncio
    async def test_connection_failure_returns_false(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        with (
            patch("llmstack.gateway.middleware.rate_limit.REDIS_URL", "redis://localhost:6379"),
            patch(
                "llmstack.gateway.middleware.rate_limit.aioredis.from_url",
                side_effect=RuntimeError("down"),
            ),
        ):
            result = await mw._ensure_redis()
        assert result is False
        assert mw._redis is None


class TestRedisCheck:
    @pytest.mark.asyncio
    async def test_success(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        mw._redis = AsyncMock()
        mw._redis.evalsha = AsyncMock(return_value=[1, 50, 0])
        mw._lua_sha = "sha"
        result = await mw._redis_check("key1")
        assert result == (True, 50, 0)

    @pytest.mark.asyncio
    async def test_falls_back_to_in_memory_on_exception(self):
        mw = RateLimitMiddleware(app=None, rate_limit="100/min")
        mw._redis = AsyncMock()
        mw._redis.evalsha = AsyncMock(side_effect=RuntimeError("boom"))
        mw._lua_sha = "sha"
        mw._redis_available = True

        allowed, remaining, retry_after = await mw._redis_check("key1")

        assert mw._redis_available is False
        assert allowed is True
