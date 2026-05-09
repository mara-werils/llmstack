"""Comprehensive tests for the AI observability system.

Covers quality scoring, trace store, quality tracker with drift detection,
A/B testing, alerts, and config schema.
"""

from __future__ import annotations


import pytest

from llmstack.observe.scoring import QualityScorer, QualityScore
from llmstack.observe.traces import Trace, TraceStore
from llmstack.observe.tracker import QualityTracker, QualityAlert
from llmstack.observe.ab_testing import ABTest, ABTestManager, ABTestResult


# ===================================================================
# Quality scoring tests
# ===================================================================

class TestQualityScorer:
    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_good_response(self, scorer):
        score = scorer.score(
            "What is Python?",
            "Python is a high-level, interpreted programming language known for "
            "its clear syntax and readability. It was created by Guido van Rossum "
            "and first released in 1991. Python supports multiple programming paradigms.",
        )
        assert score.overall > 0.5
        assert score.coherence > 0.5
        assert score.relevance > 0.3
        assert score.refusal < 0.3
        assert score.toxicity < 0.1

    def test_empty_response(self, scorer):
        score = scorer.score("What is Python?", "")
        assert score.overall == 0.0

    def test_refusal_detected(self, scorer):
        score = scorer.score(
            "Write me something",
            "I'm sorry, I cannot help with that request. As an AI, I must decline.",
        )
        assert score.refusal > 0.3

    def test_no_refusal_in_normal(self, scorer):
        score = scorer.score(
            "Hello",
            "Hello! How can I help you today?",
        )
        assert score.refusal < 0.1

    def test_repetitive_response(self, scorer):
        repeated = "This is a rather long sentence that keeps repeating over and over again. " * 20
        score = scorer.score("Tell me something", repeated)
        assert score.repetition > 0.3

    def test_non_repetitive_response(self, scorer):
        varied = (
            "Python is a language. It supports many paradigms. "
            "You can use it for web development, data science, and automation. "
            "The syntax is clean and readable."
        )
        score = scorer.score("Tell me about Python", varied)
        assert score.repetition < 0.3

    def test_relevance_high_overlap(self, scorer):
        score = scorer.score(
            "How does garbage collection work in Python?",
            "Garbage collection in Python works through reference counting "
            "and a cyclic garbage collector. When an object's reference count "
            "drops to zero, Python immediately deallocates it.",
        )
        assert score.relevance > 0.3

    def test_relevance_no_overlap(self, scorer):
        score = scorer.score(
            "How does garbage collection work in Python?",
            "The weather today is sunny with a high of 75 degrees.",
        )
        assert score.relevance < 0.3

    def test_coherence_short_response(self, scorer):
        score = scorer.score("question", "ok")
        assert score.coherence < 0.5

    def test_coherence_long_structured(self, scorer):
        response = (
            "Here's a detailed explanation:\n\n"
            "1. First, we need to understand the basics.\n"
            "2. Then, we apply the concept.\n"
            "3. Finally, we verify the results.\n\n"
            "This approach ensures correctness and reliability."
        )
        score = scorer.score("explain something", response)
        assert score.coherence > 0.7

    def test_score_to_dict(self, scorer):
        score = scorer.score("test", "test response")
        d = score.to_dict()
        assert "coherence" in d
        assert "relevance" in d
        assert "overall" in d
        assert all(isinstance(v, float) for v in d.values())


class TestQualityScore:
    def test_to_dict(self):
        score = QualityScore(
            coherence=0.8, relevance=0.7, refusal=0.0,
            toxicity=0.0, repetition=0.1, overall=0.75,
        )
        d = score.to_dict()
        assert d["coherence"] == 0.8
        assert d["overall"] == 0.75


# ===================================================================
# Trace tests
# ===================================================================

class TestTrace:
    def test_auto_id(self):
        t = Trace()
        assert len(t.id) > 0

    def test_auto_timestamp(self):
        t = Trace()
        assert t.timestamp > 0

    def test_total_tokens(self):
        t = Trace(input_tokens=100, output_tokens=50)
        assert t.total_tokens() == 150

    def test_to_dict(self):
        t = Trace(
            model="gpt-4o", provider="openai",
            input_tokens=100, output_tokens=50,
            latency_ms=250.5, cost_usd=0.001,
            quality={"overall": 0.8, "coherence": 0.9},
        )
        d = t.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["provider"] == "openai"
        assert d["latency_ms"] == 250.5
        assert d["quality"]["overall"] == 0.8

    def test_to_dict_with_error(self):
        t = Trace(error="timeout")
        d = t.to_dict()
        assert d["error"] == "timeout"


# ===================================================================
# TraceStore tests
# ===================================================================

class TestTraceStore:
    def test_add_and_recent(self):
        store = TraceStore(max_size=100)
        store.add(Trace(model="a"))
        store.add(Trace(model="b"))
        traces = store.recent(10)
        assert len(traces) == 2

    def test_max_size(self):
        store = TraceStore(max_size=5)
        for i in range(10):
            store.add(Trace(model=f"m{i}"))
        assert len(store.recent(100)) == 5
        assert store.total_count == 10

    def test_query_by_model(self):
        store = TraceStore()
        store.add(Trace(model="gpt-4o"))
        store.add(Trace(model="claude"))
        store.add(Trace(model="gpt-4o"))

        results = store.query(model="gpt-4o")
        assert len(results) == 2

    def test_query_by_provider(self):
        store = TraceStore()
        store.add(Trace(provider="openai"))
        store.add(Trace(provider="anthropic"))
        store.add(Trace(provider="openai"))

        results = store.query(provider="openai")
        assert len(results) == 2

    def test_query_with_error_filter(self):
        store = TraceStore()
        store.add(Trace(model="a", error="timeout"))
        store.add(Trace(model="b"))
        store.add(Trace(model="c", error="500"))

        with_errors = store.query(has_error=True)
        assert len(with_errors) == 2

        without_errors = store.query(has_error=False)
        assert len(without_errors) == 1

    def test_query_limit(self):
        store = TraceStore()
        for i in range(20):
            store.add(Trace(model="m"))
        results = store.query(limit=5)
        assert len(results) == 5

    def test_summary_empty(self):
        store = TraceStore()
        s = store.summary()
        assert s["total"] == 0

    def test_summary(self):
        store = TraceStore()
        store.add(Trace(model="gpt-4o", provider="openai",
                        latency_ms=100, cost_usd=0.01, input_tokens=50, output_tokens=20,
                        quality={"overall": 0.8}))
        store.add(Trace(model="claude", provider="anthropic",
                        latency_ms=200, cost_usd=0.02, input_tokens=60, output_tokens=30,
                        quality={"overall": 0.9}))

        s = store.summary()
        assert s["total"] == 2
        assert s["stored"] == 2
        assert s["avg_latency_ms"] == 150.0
        assert s["total_cost_usd"] == 0.03
        assert "gpt-4o" in s["models"]
        assert "openai" in s["providers"]
        assert s["avg_quality"]["overall"] == pytest.approx(0.85)


# ===================================================================
# QualityTracker tests
# ===================================================================

class TestQualityTracker:
    def test_record_scores(self):
        tracker = QualityTracker()
        tracker.record({"overall": 0.8, "coherence": 0.9}, model="gpt-4o")

        s = tracker.summary()
        assert "overall" in s["global"]
        assert s["global"]["overall"]["mean"] == pytest.approx(0.8)

    def test_per_model_tracking(self):
        tracker = QualityTracker()
        tracker.record({"overall": 0.8}, model="gpt-4o")
        tracker.record({"overall": 0.6}, model="claude")

        s = tracker.summary()
        assert "gpt-4o" in s["by_model"]
        assert "claude" in s["by_model"]

    def test_alert_on_low_quality(self):
        tracker = QualityTracker(alert_threshold=0.5)

        # Need >= 10 samples before alerts fire
        for _ in range(15):
            tracker.record({"overall": 0.3}, model="bad-model")

        alerts = tracker.get_alerts()
        assert len(alerts) > 0
        assert any(a.metric == "overall" for a in alerts)

    def test_no_alert_above_threshold(self):
        tracker = QualityTracker(alert_threshold=0.3)

        for _ in range(15):
            tracker.record({"overall": 0.8}, model="good-model")

        alerts = tracker.get_alerts()
        # Filter out drift alerts
        threshold_alerts = [a for a in alerts if "dropped" in a.message]
        assert len(threshold_alerts) == 0

    def test_drift_detection(self):
        tracker = QualityTracker(drift_threshold=-0.1)

        # Good quality first
        for _ in range(30):
            tracker.record({"overall": 0.9}, model="m")
        # Then bad
        for _ in range(30):
            tracker.record({"overall": 0.5}, model="m")

        alerts = tracker.get_alerts()
        drift_alerts = [a for a in alerts if "drift" in a.message.lower()]
        assert len(drift_alerts) > 0

    def test_summary_format(self):
        tracker = QualityTracker()
        tracker.record({"overall": 0.8, "coherence": 0.9}, model="m")
        s = tracker.summary()
        assert "global" in s
        assert "by_model" in s
        assert "alerts" in s


class TestQualityAlert:
    def test_to_dict(self):
        alert = QualityAlert(
            metric="overall", model="gpt-4o", provider="openai",
            current_value=0.3, threshold=0.5, severity="critical",
            message="quality dropped",
        )
        d = alert.to_dict()
        assert d["metric"] == "overall"
        assert d["severity"] == "critical"

    def test_auto_timestamp(self):
        alert = QualityAlert(metric="test")
        assert alert.timestamp > 0


# ===================================================================
# A/B Testing tests
# ===================================================================

class TestABTest:
    def test_select_model_deterministic(self):
        test = ABTest(name="t1", model_a="gpt-4o", model_b="claude", traffic_split=0.5)
        model1 = test.select_model("req-123")
        model2 = test.select_model("req-123")
        assert model1 == model2  # same request_id → same model

    def test_select_splits_traffic(self):
        test = ABTest(name="t1", model_a="a", model_b="b", traffic_split=0.5)
        selections = {"a": 0, "b": 0}
        for i in range(100):
            model = test.select_model(f"req-{i}")
            selections[model] += 1
        # Roughly 50/50 (within reason)
        assert selections["a"] > 20
        assert selections["b"] > 20

    def test_auto_timestamp(self):
        test = ABTest(name="t1", model_a="a", model_b="b")
        assert test.created_at > 0


class TestABTestManager:
    @pytest.fixture
    def manager(self):
        m = ABTestManager()
        m.create_test(ABTest(name="test1", model_a="gpt-4o", model_b="claude"))
        return m

    def test_create_and_get(self, manager):
        test = manager.get_test("test1")
        assert test is not None
        assert test.model_a == "gpt-4o"

    def test_list_tests(self, manager):
        tests = manager.list_tests()
        assert len(tests) == 1

    def test_select_model(self, manager):
        model = manager.select_model("test1", "req-1")
        assert model in ("gpt-4o", "claude")

    def test_select_nonexistent(self, manager):
        assert manager.select_model("nonexistent") is None

    def test_record_and_results(self, manager):
        for i in range(10):
            manager.record("test1", "gpt-4o", quality=0.8, latency_ms=100, cost_usd=0.01)
        for i in range(10):
            manager.record("test1", "claude", quality=0.9, latency_ms=200, cost_usd=0.02)

        result = manager.get_results("test1")
        assert result is not None
        assert result.requests_a == 10
        assert result.requests_b == 10
        assert result.avg_quality_a == pytest.approx(0.8)
        assert result.avg_quality_b == pytest.approx(0.9)
        assert result.winner == "claude"

    def test_results_nonexistent(self, manager):
        assert manager.get_results("nonexistent") is None

    def test_stop_test(self, manager):
        manager.stop_test("test1")
        test = manager.get_test("test1")
        assert test.active is False
        assert manager.select_model("test1") is None

    def test_confidence_low_few_samples(self, manager):
        manager.record("test1", "gpt-4o", quality=0.8, latency_ms=100)
        manager.record("test1", "claude", quality=0.9, latency_ms=200)

        result = manager.get_results("test1")
        assert result.confidence == "low"

    def test_confidence_increases_with_samples(self):
        m = ABTestManager()
        m.create_test(ABTest(name="big", model_a="a", model_b="b"))

        for _ in range(100):
            m.record("big", "a", quality=0.5, latency_ms=100)
            m.record("big", "b", quality=0.7, latency_ms=100)

        result = m.get_results("big")
        assert result.confidence == "high"


class TestABTestResult:
    def test_to_dict(self):
        r = ABTestResult(
            test_name="test1", model_a="a", model_b="b",
            requests_a=50, requests_b=50,
            avg_quality_a=0.7, avg_quality_b=0.8,
            winner="b", confidence="medium",
        )
        d = r.to_dict()
        assert d["winner"] == "b"
        assert d["confidence"] == "medium"


# ===================================================================
# Config schema tests
# ===================================================================

class TestObserveConfigSchema:
    def test_defaults(self):
        from llmstack.config.schema import ObserveConfig
        config = ObserveConfig()
        assert config.quality_tracking is True
        assert config.alert_threshold == 0.4
        assert config.drift_threshold == -0.1
        assert config.trace_store_size == 5000

    def test_custom(self):
        from llmstack.config.schema import ObserveConfig
        config = ObserveConfig(
            alert_threshold=0.6,
            drift_threshold=-0.05,
            trace_store_size=10000,
        )
        assert config.alert_threshold == 0.6
        assert config.trace_store_size == 10000


# ===================================================================
# State management tests
# ===================================================================

class TestObserveState:
    def test_init_and_get(self):
        from llmstack.observe._state import init_observe, get_trace_store, get_scorer, get_tracker, get_ab_manager

        init_observe()
        assert get_trace_store() is not None
        assert get_scorer() is not None
        assert get_tracker() is not None
        assert get_ab_manager() is not None
