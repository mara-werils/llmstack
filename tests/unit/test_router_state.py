"""Tests for the router module-level singletons."""

from __future__ import annotations

from llmstack.gateway.router import _state
from llmstack.gateway.router.router import ModelRouter, ModelTier
from llmstack.gateway.router.stats import RouterStats


def _make_router() -> ModelRouter:
    return ModelRouter([ModelTier(name="llama3.2:1b", tier="simple")])


def test_accessors_none_before_init(monkeypatch):
    monkeypatch.setattr(_state, "_router", None)
    monkeypatch.setattr(_state, "_stats", None)
    assert _state.get_router() is None
    assert _state.get_stats() is None


def test_init_router_sets_singletons(monkeypatch):
    monkeypatch.setattr(_state, "_router", None)
    monkeypatch.setattr(_state, "_stats", None)
    router = _make_router()
    stats = RouterStats()

    _state.init_router(router, stats)

    assert _state.get_router() is router
    assert _state.get_stats() is stats
