"""Tests for request deduplication."""

from __future__ import annotations
from llmstack.gateway.dedup_requests import RequestDeduplicator, DedupConfig


class TestRequestDeduplicator:
    def test_generate_key(self):
        d = RequestDeduplicator()
        k1 = d.generate_key("POST", "/v1/chat", "body1")
        k2 = d.generate_key("POST", "/v1/chat", "body2")
        assert k1 != k2
        assert len(k1) == 24

    def test_cache_and_retrieve(self):
        d = RequestDeduplicator()
        k = d.generate_key("POST", "/v1/chat", "test")
        d.cache_response(k, 200, {"result": "ok"})
        cached = d.get_cached(k)
        assert cached is not None
        assert cached.body == {"result": "ok"}

    def test_miss_returns_none(self):
        d = RequestDeduplicator()
        assert d.get_cached("nonexistent") is None

    def test_clear(self):
        d = RequestDeduplicator()
        d.cache_response("k1", 200, {})
        d.cache_response("k2", 200, {})
        count = d.clear()
        assert count == 2
        assert d.get_cached("k1") is None

    def test_stats(self):
        d = RequestDeduplicator()
        stats = d.get_stats()
        assert "total_cached" in stats
        assert "max_entries" in stats

    def test_eviction(self):
        cfg = DedupConfig(max_entries=2)
        d = RequestDeduplicator(config=cfg)
        d.cache_response("k1", 200, {})
        d.cache_response("k2", 200, {})
        d.cache_response("k3", 200, {})
        assert d.get_stats()["total_cached"] <= 2
