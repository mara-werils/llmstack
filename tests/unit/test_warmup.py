"""Tests for gateway model warm-up."""

from __future__ import annotations
from llmstack.gateway.warmup import WarmupResult, WarmupManager


class TestWarmupResult:
    def test_to_dict(self):
        r = WarmupResult(model="test", success=True, latency_ms=42.0)
        d = r.to_dict()
        assert d["model"] == "test"
        assert d["success"] is True
        assert d["latency_ms"] == 42.0

    def test_default_values(self):
        r = WarmupResult(model="x")
        assert r.success is False
        assert r.error == ""


class TestWarmupManager:
    def test_init(self):
        m = WarmupManager(models=["a", "b"])
        assert len(m.models) == 2
