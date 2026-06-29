"""Tests for the Redis-backed response cache I/O paths (via fakeredis)."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from llmstack.gateway import cache as cache_mod
from llmstack.gateway.cache import CacheStats, ResponseCache

MSGS = [{"role": "user", "content": "hello"}]


@pytest.fixture
def fake_redis(monkeypatch):
    server = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cache_mod.aioredis, "from_url", lambda *a, **k: server)
    return server


@pytest.fixture
async def connected(fake_redis):
    c = ResponseCache(redis_url="redis://fake")
    await c.connect()
    return c


class TestStatsExtras:
    def test_total_requests(self):
        s = CacheStats()
        s.record_hit(1.0)
        s.record_miss()
        assert s.total_requests == 2


class TestConnect:
    async def test_connect_sets_connected(self, connected):
        assert connected.is_connected is True

    async def test_connect_noop_without_url(self):
        c = ResponseCache(redis_url="")
        await c.connect()
        assert c.is_connected is False

    async def test_connect_handles_failure(self, monkeypatch):
        class _Bad:
            async def ping(self):
                raise ConnectionError("down")

        monkeypatch.setattr(cache_mod.aioredis, "from_url", lambda *a, **k: _Bad())
        c = ResponseCache(redis_url="redis://x")
        await c.connect()
        assert c.is_connected is False

    async def test_close(self, connected):
        await connected.close()
        assert connected.is_connected is False


class TestGetPut:
    async def test_miss_when_not_connected(self):
        c = ResponseCache(redis_url="")
        assert await c.get("m", MSGS, 0.0) is None

    async def test_put_then_get_roundtrip(self, connected):
        await connected.put("m", MSGS, {"answer": 42}, 0.0)
        got = await connected.get("m", MSGS, 0.0)
        assert got["answer"] == 42
        assert got["_cached"] is True
        assert "_cache_age_s" in got
        assert connected.stats.hits == 1

    async def test_put_does_not_mutate_caller_response(self, connected):
        response = {"answer": 42}
        await connected.put("m", MSGS, response, 0.0)
        # The internal _cached_at marker must not leak into the caller's dict.
        assert "_cached_at" not in response

    async def test_get_miss_records_miss(self, connected):
        assert await connected.get("m", MSGS, 0.0) is None
        assert connected.stats.misses == 1

    async def test_high_temperature_skips(self, connected):
        await connected.put("m", MSGS, {"x": 1}, 0.9)
        assert await connected.get("m", MSGS, 0.9) is None
        # skipped lookups still count as a miss
        assert connected.stats.misses == 1

    async def test_get_handles_bad_json(self, connected, fake_redis):
        key = ResponseCache._build_cache_key("m", MSGS, 0.0)
        await fake_redis.set(key, "{not json")
        assert await connected.get("m", MSGS, 0.0) is None
        assert connected.stats.misses == 1

    async def test_cache_age_none_when_cached_at_missing(self, connected, fake_redis):
        import json

        # A legacy/externally-written entry with no _cached_at marker.
        key = ResponseCache._build_cache_key("m", MSGS, 0.0)
        await fake_redis.set(key, json.dumps({"answer": 1}))
        got = await connected.get("m", MSGS, 0.0)
        assert got["answer"] == 1
        # Unknown age must be None, not ~55 years (time.time() - 0).
        assert got["_cache_age_s"] is None

    async def test_cache_age_is_small_for_fresh_put(self, connected):
        await connected.put("m", MSGS, {"answer": 2}, 0.0)
        got = await connected.get("m", MSGS, 0.0)
        assert got["_cache_age_s"] is not None
        assert 0 <= got["_cache_age_s"] < 5


class TestInvalidate:
    async def test_invalidate_deletes_matching(self, connected):
        await connected.put("m1", MSGS, {"a": 1}, 0.0)
        await connected.put("m2", MSGS, {"b": 2}, 0.0)
        deleted = await connected.invalidate("*")
        assert deleted == 2

    async def test_invalidate_no_keys(self, connected):
        assert await connected.invalidate("*") == 0

    async def test_invalidate_not_connected(self):
        c = ResponseCache(redis_url="")
        assert await c.invalidate("*") == 0


class TestSingleton:
    async def test_get_cache_returns_singleton(self, fake_redis, monkeypatch):
        monkeypatch.setattr(cache_mod, "_cache", None)
        monkeypatch.setattr(cache_mod, "REDIS_URL", "redis://fake")
        c1 = await cache_mod.get_cache()
        c2 = await cache_mod.get_cache()
        assert c1 is c2
