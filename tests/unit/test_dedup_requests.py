"""Tests for request deduplication."""

from __future__ import annotations

import time

import pytest

from llmstack.gateway.dedup_requests import (
    DedupConfig,
    RequestDeduplicator,
)


@pytest.fixture
def dedup():
    return RequestDeduplicator()


class TestRequestDeduplicator:
    def test_no_cached_initially(self, dedup):
        assert dedup.get_cached("key1") is None

    def test_cache_and_retrieve(self, dedup):
        dedup.cache_response("key1", 200, {"result": "ok"})
        cached = dedup.get_cached("key1")
        assert cached is not None
        assert cached.status_code == 200
        assert cached.body == {"result": "ok"}

    def test_ttl_expiry(self):
        config = DedupConfig(ttl=0.01)  # 10ms TTL
        dedup = RequestDeduplicator(config)
        dedup.cache_response("key1", 200, {"data": "test"})
        time.sleep(0.02)
        assert dedup.get_cached("key1") is None

    def test_generate_key_deterministic(self, dedup):
        k1 = dedup.generate_key("POST", "/v1/chat", "body1")
        k2 = dedup.generate_key("POST", "/v1/chat", "body1")
        assert k1 == k2

    def test_different_requests_different_keys(self, dedup):
        k1 = dedup.generate_key("POST", "/v1/chat", "body1")
        k2 = dedup.generate_key("POST", "/v1/chat", "body2")
        assert k1 != k2

    def test_max_entries_enforced(self):
        config = DedupConfig(max_entries=3)
        dedup = RequestDeduplicator(config)
        for i in range(5):
            dedup.cache_response(f"key{i}", 200, {"i": i})
        stats = dedup.get_stats()
        assert stats["total_cached"] <= 3

    def test_clear(self, dedup):
        dedup.cache_response("k1", 200, {})
        dedup.cache_response("k2", 200, {})
        removed = dedup.clear()
        assert removed == 2
        assert dedup.get_cached("k1") is None

    def test_stats(self, dedup):
        dedup.cache_response("k1", 200, {})
        stats = dedup.get_stats()
        assert stats["total_cached"] == 1
        assert "ttl_seconds" in stats
        assert "max_entries" in stats

    def test_stats_includes_hit_metrics(self, dedup):
        dedup.cache_response("k1", 200, {})
        dedup.get_cached("k1")  # hit
        dedup.get_cached("missing")  # miss
        stats = dedup.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_expired_entries_purged_on_write(self):
        config = DedupConfig(ttl=0.01)
        dedup = RequestDeduplicator(config)
        dedup.cache_response("old", 200, {})
        time.sleep(0.02)  # let "old" expire
        dedup.cache_response("new", 200, {})
        # Writing a new entry sweeps the expired one out instead of leaving it.
        assert dedup.cache_size == 1
        assert dedup.get_cached("new") is not None

    def test_overwrite_cached(self, dedup):
        dedup.cache_response("k1", 200, {"v": 1})
        dedup.cache_response("k1", 201, {"v": 2})
        cached = dedup.get_cached("k1")
        assert cached.status_code == 201
        assert cached.body == {"v": 2}

    def test_hit_rate(self, dedup):
        assert dedup.hit_rate == 0.0
        dedup.cache_response("k1", 200, {})
        dedup.get_cached("k1")  # hit
        dedup.get_cached("k2")  # miss
        assert dedup.hit_rate == 0.5

    def test_cache_size(self, dedup):
        assert dedup.cache_size == 0
        dedup.cache_response("k1", 200, {})
        dedup.cache_response("k2", 200, {})
        assert dedup.cache_size == 2
