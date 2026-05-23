"""Tests for model alias mapping."""

from __future__ import annotations

import pytest

from llmstack.gateway.model_aliases import (
    AliasConfig,
    ModelAliasResolver,
    DEFAULT_ALIASES,
)


@pytest.fixture
def resolver():
    return ModelAliasResolver()


class TestModelAliasResolver:
    def test_resolve_default_alias(self, resolver):
        assert resolver.resolve("fast") == "llama3.2:1b"
        assert resolver.resolve("smart") == "llama3.1:8b"

    def test_resolve_unknown_passes_through(self, resolver):
        assert resolver.resolve("gpt-4o") == "gpt-4o"

    def test_case_insensitive(self, resolver):
        assert resolver.resolve("FAST") == "llama3.2:1b"
        assert resolver.resolve("Fast") == "llama3.2:1b"

    def test_add_custom_alias(self, resolver):
        resolver.add_alias("my-model", "custom/model:v1")
        assert resolver.resolve("my-model") == "custom/model:v1"

    def test_remove_alias(self, resolver):
        resolver.add_alias("temp", "some-model")
        assert resolver.remove_alias("temp") is True
        assert resolver.resolve("temp") == "temp"

    def test_remove_nonexistent(self, resolver):
        assert resolver.remove_alias("nope") is False

    def test_list_aliases(self, resolver):
        aliases = resolver.list_aliases()
        assert "fast" in aliases
        assert "smart" in aliases

    def test_custom_config_no_defaults(self):
        config = AliasConfig(
            include_defaults=False,
            custom_aliases={"mine": "my-model:1b"},
        )
        resolver = ModelAliasResolver(config=config)
        assert resolver.resolve("fast") == "fast"  # No default
        assert resolver.resolve("mine") == "my-model:1b"

    def test_custom_overrides_default(self):
        config = AliasConfig(custom_aliases={"fast": "my-fast-model"})
        resolver = ModelAliasResolver(config=config)
        assert resolver.resolve("fast") == "my-fast-model"

    def test_stats(self, resolver):
        stats = resolver.get_stats()
        assert stats["total_aliases"] == len(DEFAULT_ALIASES)
        assert "aliases" in stats
