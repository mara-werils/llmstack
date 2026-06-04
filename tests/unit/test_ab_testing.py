"""Comprehensive tests for A/B testing — traffic splitting, metrics, results."""

from __future__ import annotations

import pytest

from llmstack.observe.ab_testing import (
    ABTest,
    ABTestManager,
    ABTestResult,
    _determine_winner,
    _ModelMetrics,
)


# ---------------------------------------------------------------------------
# ABTestResult
# ---------------------------------------------------------------------------


class TestABTestResult:
    def test_defaults(self) -> None:
        r = ABTestResult()
        assert r.requests_a == 0
        assert r.winner == ""

    def test_to_dict(self) -> None:
        r = ABTestResult(
            test_name="test1",
            model_a="gpt-4",
            model_b="llama",
            requests_a=10,
            requests_b=15,
            avg_quality_a=0.85123,
            avg_quality_b=0.78456,
            avg_latency_a_ms=100.789,
            avg_latency_b_ms=200.123,
            avg_cost_a_usd=0.001234,
            avg_cost_b_usd=0.002345,
            winner="gpt-4",
            confidence="high",
        )
        d = r.to_dict()
        assert d["test_name"] == "test1"
        assert d["avg_quality_a"] == 0.8512  # rounded to 4 decimal places
        assert d["avg_latency_a_ms"] == 100.8  # rounded to 1 decimal place
        assert d["avg_cost_a_usd"] == 0.001234  # rounded to 6 decimal places


# ---------------------------------------------------------------------------
# ABTest
# ---------------------------------------------------------------------------


class TestABTest:
    def test_defaults(self) -> None:
        t = ABTest(name="test1", model_a="a", model_b="b")
        assert t.traffic_split == 0.5
        assert t.active is True
        assert t.created_at > 0

    def test_select_model_deterministic(self) -> None:
        t = ABTest(name="test1", model_a="a", model_b="b")
        # Same request_id should always return same model
        result1 = t.select_model("req-123")
        result2 = t.select_model("req-123")
        assert result1 == result2
        assert result1 in ("a", "b")

    def test_select_model_without_request_id(self) -> None:
        t = ABTest(name="test1", model_a="a", model_b="b")
        result = t.select_model()
        assert result in ("a", "b")

    def test_traffic_split_all_a(self) -> None:
        t = ABTest(name="test1", model_a="a", model_b="b", traffic_split=0.0)
        # With traffic_split=0.0, bucket < 0.0 is never true, so always model_a
        results = {t.select_model(f"req-{i}") for i in range(100)}
        assert results == {"a"}

    def test_traffic_split_all_b(self) -> None:
        t = ABTest(name="test1", model_a="a", model_b="b", traffic_split=1.0)
        results = {t.select_model(f"req-{i}") for i in range(100)}
        assert results == {"b"}

    def test_traffic_split_roughly_even(self) -> None:
        t = ABTest(name="test1", model_a="a", model_b="b", traffic_split=0.5)
        counts = {"a": 0, "b": 0}
        for i in range(1000):
            model = t.select_model(f"request-{i}")
            counts[model] += 1
        # Roughly 50/50 — allow wide margin
        assert counts["a"] > 200
        assert counts["b"] > 200


# ---------------------------------------------------------------------------
# _determine_winner
# ---------------------------------------------------------------------------


class TestDetermineWinner:
    def test_empty_both(self) -> None:
        winner, confidence = _determine_winner([], [], "a", "b")
        assert winner == ""
        assert confidence == "low"

    def test_few_samples_low_confidence(self) -> None:
        winner, confidence = _determine_winner([0.9, 0.8], [0.7, 0.6], "a", "b")
        assert winner == "a"
        assert confidence == "low"

    def test_medium_confidence(self) -> None:
        vals_a = [0.8] * 30
        vals_b = [0.7] * 30  # diff = 0.1 > 0.03, n = 30
        winner, confidence = _determine_winner(vals_a, vals_b, "a", "b")
        assert winner == "a"
        assert confidence == "medium"

    def test_high_confidence(self) -> None:
        vals_a = [0.9] * 100
        vals_b = [0.7] * 100  # diff = 0.2 > 0.05, n = 100
        winner, confidence = _determine_winner(vals_a, vals_b, "a", "b")
        assert winner == "a"
        assert confidence == "high"

    def test_b_wins(self) -> None:
        vals_a = [0.5] * 10
        vals_b = [0.9] * 10
        winner, _ = _determine_winner(vals_a, vals_b, "a", "b")
        assert winner == "b"

    def test_tie_goes_to_a(self) -> None:
        vals_a = [0.8] * 10
        vals_b = [0.8] * 10
        winner, _ = _determine_winner(vals_a, vals_b, "a", "b")
        assert winner == "a"  # mean_a >= mean_b

    def test_low_confidence_small_diff(self) -> None:
        vals_a = [0.80] * 100
        vals_b = [0.79] * 100  # diff = 0.01 < 0.05
        winner, confidence = _determine_winner(vals_a, vals_b, "a", "b")
        assert confidence == "low"

    def test_unequal_sample_sizes(self) -> None:
        vals_a = [0.9] * 3
        vals_b = [0.7] * 50
        winner, confidence = _determine_winner(vals_a, vals_b, "a", "b")
        # min(3, 50) = 3 < 5 → low confidence
        assert confidence == "low"


# ---------------------------------------------------------------------------
# ABTestManager
# ---------------------------------------------------------------------------


class TestABTestManager:
    @pytest.fixture
    def manager(self) -> ABTestManager:
        return ABTestManager()

    @pytest.fixture
    def test_with_manager(self, manager: ABTestManager) -> ABTest:
        test = ABTest(name="exp1", model_a="gpt-4", model_b="llama")
        manager.create_test(test)
        return test

    def test_create_test(self, manager: ABTestManager) -> None:
        test = ABTest(name="exp1", model_a="a", model_b="b")
        manager.create_test(test)
        assert manager.get_test("exp1") is test

    def test_get_test_nonexistent(self, manager: ABTestManager) -> None:
        assert manager.get_test("nope") is None

    def test_list_tests(self, manager: ABTestManager) -> None:
        manager.create_test(ABTest(name="t1", model_a="a", model_b="b"))
        manager.create_test(ABTest(name="t2", model_a="c", model_b="d"))
        tests = manager.list_tests()
        assert len(tests) == 2

    def test_select_model(self, manager: ABTestManager, test_with_manager: ABTest) -> None:
        model = manager.select_model("exp1", "req-1")
        assert model in ("gpt-4", "llama")

    def test_select_model_nonexistent_test(self, manager: ABTestManager) -> None:
        assert manager.select_model("nope") is None

    def test_select_model_inactive_test(self, manager: ABTestManager) -> None:
        test = ABTest(name="inactive", model_a="a", model_b="b", active=False)
        manager.create_test(test)
        assert manager.select_model("inactive") is None

    def test_record_and_get_results(
        self, manager: ABTestManager, test_with_manager: ABTest
    ) -> None:
        manager.record("exp1", "gpt-4", quality=0.9, latency_ms=100, cost_usd=0.01)
        manager.record("exp1", "gpt-4", quality=0.8, latency_ms=120, cost_usd=0.01)
        manager.record("exp1", "llama", quality=0.7, latency_ms=80, cost_usd=0.001)

        result = manager.get_results("exp1")
        assert result is not None
        assert result.requests_a == 2
        assert result.requests_b == 1
        assert result.avg_quality_a == pytest.approx(0.85, abs=0.01)
        assert result.avg_quality_b == pytest.approx(0.7, abs=0.01)

    def test_record_nonexistent_test(self, manager: ABTestManager) -> None:
        # Should not raise
        manager.record("nope", "a", quality=0.5, latency_ms=50)

    def test_record_nonexistent_model(
        self, manager: ABTestManager, test_with_manager: ABTest
    ) -> None:
        # Should not raise — model not in test
        manager.record("exp1", "unknown-model", quality=0.5, latency_ms=50)

    def test_get_results_nonexistent(self, manager: ABTestManager) -> None:
        assert manager.get_results("nope") is None

    def test_get_results_no_data(self, manager: ABTestManager, test_with_manager: ABTest) -> None:
        result = manager.get_results("exp1")
        assert result is not None
        assert result.requests_a == 0
        assert result.requests_b == 0

    def test_stop_test(self, manager: ABTestManager, test_with_manager: ABTest) -> None:
        manager.stop_test("exp1")
        test = manager.get_test("exp1")
        assert test.active is False
        # select_model should return None for inactive test
        assert manager.select_model("exp1") is None

    def test_stop_nonexistent_test(self, manager: ABTestManager) -> None:
        # Should not raise
        manager.stop_test("nope")

    def test_results_winner_determination(self, manager: ABTestManager) -> None:
        test = ABTest(name="winner_test", model_a="fast", model_b="slow")
        manager.create_test(test)
        for _ in range(30):
            manager.record("winner_test", "fast", quality=0.9, latency_ms=50)
            manager.record("winner_test", "slow", quality=0.7, latency_ms=200)

        result = manager.get_results("winner_test")
        assert result.winner == "fast"
        assert result.confidence in ("medium", "high")


# ---------------------------------------------------------------------------
# _ModelMetrics
# ---------------------------------------------------------------------------


class TestModelMetrics:
    def test_defaults(self) -> None:
        m = _ModelMetrics()
        assert m.qualities == []
        assert m.latencies == []
        assert m.costs == []

    def test_append(self) -> None:
        m = _ModelMetrics()
        m.qualities.append(0.9)
        m.latencies.append(100.0)
        m.costs.append(0.01)
        assert len(m.qualities) == 1
