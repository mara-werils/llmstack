"""Tests for model performance leaderboard."""

import pytest

from llmstack.gateway.leaderboard import Leaderboard, ModelMetrics


@pytest.fixture
def lb():
    board = Leaderboard()
    # Populate with test data
    for _ in range(10):
        board.record("gpt-4o", "openai", latency_ms=500, tokens=100,
                      cost_usd=0.01, quality_score=0.9)
    for _ in range(10):
        board.record("llama3.2", "local", latency_ms=200, tokens=80,
                      cost_usd=0.0, quality_score=0.7)
    for _ in range(10):
        board.record("claude-sonnet", "anthropic", latency_ms=400, tokens=120,
                      cost_usd=0.005, quality_score=0.95)
    return board


class TestModelMetrics:
    def test_record_and_avg(self):
        m = ModelMetrics(model="test")
        m.record(latency_ms=100, tokens=50, quality_score=0.8)
        m.record(latency_ms=200, tokens=50, quality_score=0.9)
        assert m.total_requests == 2
        assert m.avg_latency_ms == 150.0
        assert abs(m.avg_quality - 0.85) < 1e-9

    def test_percentiles(self):
        m = ModelMetrics(model="test")
        for i in range(100):
            m.record(latency_ms=float(i + 1))
        assert 50.0 <= m.p50_latency_ms <= 51.0
        assert 95.0 <= m.p95_latency_ms <= 96.0
        assert 99.0 <= m.p99_latency_ms <= 100.0

    def test_error_rate(self):
        m = ModelMetrics(model="test")
        m.record(latency_ms=100, error=False)
        m.record(latency_ms=100, error=True)
        assert m.error_rate == 0.5

    def test_cost_per_request(self):
        m = ModelMetrics(model="test")
        m.record(latency_ms=100, cost_usd=0.10)
        m.record(latency_ms=100, cost_usd=0.20)
        assert abs(m.avg_cost_per_request - 0.15) < 1e-9

    def test_to_dict(self):
        m = ModelMetrics(model="test", provider="local")
        m.record(latency_ms=100, tokens=50, quality_score=0.8)
        d = m.to_dict()
        assert d["model"] == "test"
        assert d["total_requests"] == 1
        assert "p50_latency_ms" in d


class TestLeaderboard:
    def test_rankings_by_quality(self, lb):
        rankings = lb.get_rankings(sort_by="quality")
        assert rankings[0]["model"] == "claude-sonnet"
        assert rankings[0]["rank"] == 1

    def test_rankings_by_latency(self, lb):
        rankings = lb.get_rankings(sort_by="latency")
        assert rankings[0]["model"] == "llama3.2"

    def test_rankings_by_cost(self, lb):
        rankings = lb.get_rankings(sort_by="cost")
        assert rankings[0]["model"] == "llama3.2"  # free

    def test_min_requests_filter(self, lb):
        lb.record("rare-model", "test", latency_ms=100)  # only 1 request
        rankings = lb.get_rankings(min_requests=5)
        model_names = [r["model"] for r in rankings]
        assert "rare-model" not in model_names

    def test_compare(self, lb):
        comparison = lb.compare(["gpt-4o", "llama3.2"])
        assert len(comparison) == 2
        names = {c["model"] for c in comparison}
        assert names == {"gpt-4o", "llama3.2"}

    def test_get_model(self, lb):
        result = lb.get_model("gpt-4o")
        assert result is not None
        assert result["total_requests"] == 10

    def test_get_nonexistent_model(self, lb):
        assert lb.get_model("nonexistent") is None

    def test_summary(self, lb):
        summary = lb.get_summary()
        assert summary["total_models"] == 3
        assert summary["total_requests"] == 30
        assert len(summary["top_by_quality"]) <= 3
