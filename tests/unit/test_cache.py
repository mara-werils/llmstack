"""Tests for semantic response cache."""

from __future__ import annotations

import pytest

from llmstack.gateway.cache import ResponseCache, CacheStats


class TestCacheKeyGeneration:
    """Test deterministic cache key generation."""

    def test_same_input_same_key(self):
        key1 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.0)
        key2 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.0)
        assert key1 == key2
        assert key1 != ""

    def test_different_model_different_key(self):
        key1 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.0)
        key2 = ResponseCache._build_cache_key("mistral", [{"role": "user", "content": "hello"}], 0.0)
        assert key1 != key2

    def test_different_message_different_key(self):
        key1 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.0)
        key2 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "world"}], 0.0)
        assert key1 != key2

    def test_high_temperature_skips_cache(self):
        """Responses with temperature > 0.1 should not be cached (non-deterministic)."""
        key = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.9)
        assert key == ""

    def test_low_temperature_uses_cache(self):
        key = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.0)
        assert key != ""
        key2 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hello"}], 0.1)
        assert key2 != ""

    def test_message_order_matters(self):
        msgs1 = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        msgs2 = [{"role": "assistant", "content": "b"}, {"role": "user", "content": "a"}]
        key1 = ResponseCache._build_cache_key("llama3", msgs1, 0.0)
        key2 = ResponseCache._build_cache_key("llama3", msgs2, 0.0)
        assert key1 != key2

    def test_extra_fields_in_messages_ignored(self):
        """Only role and content matter for cache key."""
        key1 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hi"}], 0.0)
        key2 = ResponseCache._build_cache_key("llama3", [{"role": "user", "content": "hi", "name": "bob"}], 0.0)
        assert key1 == key2


class TestCacheStats:
    """Test cache statistics tracking."""

    def test_initial_stats(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats()
        stats.record_hit(1.0)
        stats.record_hit(2.0)
        stats.record_miss()
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_avg_latency(self):
        stats = CacheStats()
        stats.record_hit(10.0)
        stats.record_hit(20.0)
        assert stats.avg_hit_latency_ms == 15.0

    def test_to_dict(self):
        stats = CacheStats()
        stats.record_hit(5.0)
        stats.record_miss()
        d = stats.to_dict()
        assert d["hits"] == 1
        assert d["misses"] == 1
        assert d["hit_rate"] == 0.5
