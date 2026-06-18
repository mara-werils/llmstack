"""Tests for prompt prefix caching."""

import pytest

from llmstack.gateway.prompt_cache import PromptPrefixCache


@pytest.fixture
def cache():
    return PromptPrefixCache(max_entries=10, min_prefix_length=10)


SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant with extensive knowledge."}
USER_MSG = {"role": "user", "content": "What is Python?"}


class TestPromptPrefixCache:
    def test_store_and_lookup(self, cache):
        messages = [SYSTEM_MSG, USER_MSG]
        cache.store(messages, token_count=50)
        result = cache.lookup(messages)
        assert result is not None
        assert result.hit_count == 1

    def test_same_prefix_different_query(self, cache):
        msgs1 = [SYSTEM_MSG, {"role": "user", "content": "Question A"}]
        msgs2 = [SYSTEM_MSG, {"role": "user", "content": "Question B"}]

        cache.store(msgs1, token_count=50)
        result = cache.lookup(msgs2)
        # Same system prompt prefix -> cache hit
        assert result is not None

    def test_different_prefix_miss(self, cache):
        msgs1 = [{"role": "system", "content": "You are a coding assistant."}, USER_MSG]
        msgs2 = [{"role": "system", "content": "You are a math tutor."}, USER_MSG]

        cache.store(msgs1)
        result = cache.lookup(msgs2)
        assert result is None

    def test_miss_returns_none(self, cache):
        result = cache.lookup([USER_MSG])
        assert result is None

    def test_hit_count_increments(self, cache):
        msgs = [SYSTEM_MSG, USER_MSG]
        cache.store(msgs)
        cache.lookup(msgs)
        cache.lookup(msgs)
        result = cache.lookup(msgs)
        assert result.hit_count == 3

    def test_lru_eviction(self):
        small_cache = PromptPrefixCache(max_entries=2, min_prefix_length=5)
        for i in range(3):
            msgs = [
                {"role": "system", "content": f"System prompt number {i} is long enough"},
                {"role": "user", "content": "test"},
            ]
            small_cache.store(msgs)

        stats = small_cache.get_stats()
        assert stats["total_entries"] <= 2

    def test_invalidate(self, cache):
        msgs = [SYSTEM_MSG, USER_MSG]
        entry = cache.store(msgs)
        assert cache.invalidate(entry.hash) is True
        assert cache.lookup(msgs) is None

    def test_clear(self, cache):
        cache.store([SYSTEM_MSG, USER_MSG])
        count = cache.clear()
        assert count == 1
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_stats(self, cache):
        msgs = [SYSTEM_MSG, USER_MSG]
        cache.store(msgs)
        cache.lookup(msgs)  # hit
        cache.lookup([{"role": "user", "content": "x"}])  # miss

        stats = cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] >= 1
        assert stats["hit_rate"] > 0

    def test_compute_prefix_hash_deterministic(self):
        msgs = [SYSTEM_MSG, USER_MSG]
        h1 = PromptPrefixCache.compute_prefix_hash(msgs)
        h2 = PromptPrefixCache.compute_prefix_hash(msgs)
        assert h1 == h2
        assert len(h1) == 16

    def test_compute_prefix_hash_empty_messages(self):
        assert PromptPrefixCache.compute_prefix_hash([]) == ""

    def test_compute_prefix_hash_explicit_length(self):
        msgs = [SYSTEM_MSG, USER_MSG, {"role": "assistant", "content": "hi"}]
        h = PromptPrefixCache.compute_prefix_hash(msgs, prefix_length=1)
        assert len(h) == 16

    def test_hit_rate_and_size_properties(self, cache):
        assert cache.hit_rate == 0.0
        assert cache.size == 0
        cache.store([SYSTEM_MSG, USER_MSG])
        cache.lookup([SYSTEM_MSG, USER_MSG])
        assert cache.hit_rate == 1.0
        assert cache.size == 1

    def test_store_empty_messages_returns_none(self, cache):
        assert cache.store([]) is None

    def test_store_short_prefix_returns_none(self, cache):
        big_cache = PromptPrefixCache(max_entries=10, min_prefix_length=500)
        result = big_cache.store([SYSTEM_MSG, USER_MSG])
        assert result is None

    def test_store_existing_prefix_returns_cached_entry(self, cache):
        msgs = [SYSTEM_MSG, USER_MSG]
        first = cache.store(msgs)
        second = cache.store(msgs)
        assert second.hash == first.hash
        assert cache.size == 1
