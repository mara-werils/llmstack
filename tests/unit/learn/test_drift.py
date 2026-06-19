"""Tests for drift detection."""

from __future__ import annotations

import time

import pytest

from llmstack.learn.drift import (
    DriftAlert,
    DriftConfig,
    DriftDetector,
)
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def detector(store):
    config = DriftConfig(
        min_baseline_samples=5,
        min_recent_samples=2,
    )
    return DriftDetector(store=store, config=config)


def _add_feedback(
    store: FeedbackStore,
    *,
    query: str = "hello world",
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP,
    age_seconds: float = 0.0,
    fid: str | None = None,
) -> Feedback:
    """Insert a Feedback row with a timestamp `age_seconds` in the past."""
    fb = Feedback(
        query=query,
        feedback_type=feedback_type,
        timestamp=time.time() - age_seconds,
    )
    if fid is not None:
        fb.id = fid
    store.add_feedback(fb)
    return fb


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDriftAlert:
    def test_defaults(self):
        alert = DriftAlert(
            drift_type="topic",
            severity="medium",
            description="something shifted",
        )
        assert alert.drift_type == "topic"
        assert alert.severity == "medium"
        assert alert.description == "something shifted"
        assert isinstance(alert.timestamp, float)
        assert alert.details == {}

    def test_with_details(self):
        alert = DriftAlert(
            drift_type="quality",
            severity="high",
            description="bad",
            details={"k": 1},
        )
        assert alert.details == {"k": 1}


class TestDriftConfig:
    def test_defaults(self):
        cfg = DriftConfig()
        assert cfg.baseline_window == 604800
        assert cfg.recent_window == 86400
        assert cfg.distribution_threshold == 0.3
        assert cfg.topic_threshold == 0.4
        assert cfg.feedback_shift_threshold == 0.2
        assert cfg.min_baseline_samples == 20
        assert cfg.min_recent_samples == 5

    def test_override(self):
        cfg = DriftConfig(min_baseline_samples=1, distribution_threshold=0.9)
        assert cfg.min_baseline_samples == 1
        assert cfg.distribution_threshold == 0.9


# ---------------------------------------------------------------------------
# DriftDetector construction & properties
# ---------------------------------------------------------------------------


class TestDriftDetectorInit:
    def test_default_config(self, store):
        det = DriftDetector(store=store)
        assert det.config is not None
        assert isinstance(det.config, DriftConfig)
        assert det.store is store

    def test_custom_config(self, store):
        cfg = DriftConfig(min_baseline_samples=3)
        det = DriftDetector(store=store, config=cfg)
        assert det.config is cfg

    def test_alert_count_starts_zero(self, detector):
        assert detector.alert_count == 0

    def test_last_alert_none_initially(self, detector):
        assert detector.last_alert is None


# ---------------------------------------------------------------------------
# check() — guard rails
# ---------------------------------------------------------------------------


class TestCheckGuards:
    def test_no_data_returns_empty(self, detector):
        assert detector.check() == []

    def test_insufficient_baseline(self, store, detector):
        # Fewer than min_baseline_samples (5) total feedback rows.
        for _ in range(3):
            _add_feedback(store, age_seconds=100000)
        assert detector.check() == []

    def test_insufficient_recent(self, store):
        # Enough baseline, but recent window (default 86400) too sparse.
        config = DriftConfig(min_baseline_samples=5, min_recent_samples=5)
        det = DriftDetector(store=store, config=config)
        # Baseline rows older than recent window.
        for _ in range(10):
            _add_feedback(store, age_seconds=200000)
        # Only 1 recent row (< min_recent_samples).
        _add_feedback(store, age_seconds=10)
        assert det.check() == []

    def test_returns_list(self, store, detector):
        for _ in range(10):
            _add_feedback(store, age_seconds=200000)
        for _ in range(5):
            _add_feedback(store, age_seconds=10)
        assert isinstance(detector.check(), list)


# ---------------------------------------------------------------------------
# Query distribution drift
# ---------------------------------------------------------------------------


class TestQueryDistribution:
    def test_no_drift_same_lengths(self, store, detector):
        # Baseline and recent identical query lengths → no query alert.
        for _ in range(10):
            _add_feedback(store, query="x" * 30, age_seconds=200000)
        for _ in range(5):
            _add_feedback(store, query="x" * 30, age_seconds=10)
        alerts = detector.check()
        assert not [a for a in alerts if a.drift_type == "query_distribution"]

    def test_drift_detected_on_length_shift(self, store, detector):
        # Baseline: short queries. Recent: long queries → KL divergence high.
        for _ in range(20):
            _add_feedback(store, query="hi", age_seconds=200000)
        for _ in range(10):
            _add_feedback(store, query="z" * 300, age_seconds=10)
        alerts = detector.check()
        qa = [a for a in alerts if a.drift_type == "query_distribution"]
        assert len(qa) == 1
        assert qa[0].severity in ("medium", "high")
        assert "kl_divergence" in qa[0].details

    def test_empty_queries_skipped(self, detector):
        # Direct call: empty queries on both sides → None.
        baseline = [Feedback(query="")]
        recent = [Feedback(query="")]
        assert detector._check_query_distribution(baseline, recent) is None

    def test_high_severity_on_large_divergence(self, detector):
        # Extreme bin separation forces kl_div >= 0.6 → "high".
        baseline = [Feedback(query="x" * 10) for _ in range(20)]
        recent = [Feedback(query="x" * 400) for _ in range(20)]
        alert = detector._check_query_distribution(baseline, recent)
        assert alert is not None
        assert alert.severity == "high"


# ---------------------------------------------------------------------------
# Topic drift
# ---------------------------------------------------------------------------


class TestTopicDrift:
    def test_no_topics_returns_none(self, detector):
        # Queries with no recognized keywords → empty topic counters.
        baseline = [Feedback(query="zzz qqq") for _ in range(5)]
        recent = [Feedback(query="lorem ipsum") for _ in range(5)]
        assert detector._check_topic_drift(baseline, recent) is None

    def test_new_topic_emerges(self, detector):
        # Baseline only about python; recent introduces rust strongly.
        baseline = [Feedback(query="python question") for _ in range(50)]
        recent = [Feedback(query="rust lifetime issue") for _ in range(10)]
        alert = detector._check_topic_drift(baseline, recent)
        assert alert is not None
        assert alert.drift_type == "topic"
        assert alert.severity == "medium"
        assert "rust" in alert.details["new_topics"]

    def test_no_new_topic_when_stable(self, detector):
        # Same topic dominates both windows → no new topic alert.
        baseline = [Feedback(query="python code") for _ in range(20)]
        recent = [Feedback(query="python code") for _ in range(10)]
        assert detector._check_topic_drift(baseline, recent) is None

    def test_topic_drift_via_check(self, store, detector):
        for _ in range(30):
            _add_feedback(store, query="python script", age_seconds=200000)
        for _ in range(10):
            _add_feedback(store, query="docker container", age_seconds=10)
        alerts = detector.check()
        topic_alerts = [a for a in alerts if a.drift_type == "topic"]
        assert len(topic_alerts) == 1
        assert "docker" in topic_alerts[0].details["new_topics"]


# ---------------------------------------------------------------------------
# Feedback pattern drift
# ---------------------------------------------------------------------------


class TestFeedbackPattern:
    def test_empty_returns_none(self, detector):
        assert detector._check_feedback_pattern([], []) is None

    def test_negative_shift_detected(self, detector):
        # Baseline all positive; recent all negative → big shift.
        baseline = [Feedback(feedback_type=FeedbackType.THUMBS_UP) for _ in range(20)]
        recent = [Feedback(feedback_type=FeedbackType.THUMBS_DOWN) for _ in range(10)]
        alert = detector._check_feedback_pattern(baseline, recent)
        assert alert is not None
        assert alert.drift_type == "feedback_pattern"
        assert alert.severity == "high"  # shift 1.0 > 0.4
        assert alert.details["shift"] == 1.0

    def test_medium_severity_on_moderate_shift(self, detector):
        # Baseline 0% negative, recent 30% negative → shift 0.3 (medium).
        baseline = [Feedback(feedback_type=FeedbackType.THUMBS_UP) for _ in range(10)]
        recent = [Feedback(feedback_type=FeedbackType.THUMBS_UP) for _ in range(7)]
        recent += [Feedback(feedback_type=FeedbackType.REGENERATE) for _ in range(3)]
        alert = detector._check_feedback_pattern(baseline, recent)
        assert alert is not None
        assert alert.severity == "medium"

    def test_no_shift_when_stable(self, detector):
        baseline = [Feedback(feedback_type=FeedbackType.THUMBS_UP) for _ in range(10)]
        recent = [Feedback(feedback_type=FeedbackType.THUMBS_UP) for _ in range(5)]
        assert detector._check_feedback_pattern(baseline, recent) is None

    def test_regenerate_counts_as_negative(self, detector):
        baseline = [Feedback(feedback_type=FeedbackType.THUMBS_UP) for _ in range(10)]
        recent = [Feedback(feedback_type=FeedbackType.REGENERATE) for _ in range(10)]
        alert = detector._check_feedback_pattern(baseline, recent)
        assert alert is not None
        assert alert.details["recent_negative_rate"] == 1.0

    def test_feedback_pattern_via_check(self, store):
        config = DriftConfig(min_baseline_samples=5, min_recent_samples=2)
        det = DriftDetector(store=store, config=config)
        for _ in range(20):
            _add_feedback(store, feedback_type=FeedbackType.THUMBS_UP, age_seconds=200000)
        for _ in range(10):
            _add_feedback(store, feedback_type=FeedbackType.THUMBS_DOWN, age_seconds=10)
        alerts = det.check()
        fb_alerts = [a for a in alerts if a.drift_type == "feedback_pattern"]
        assert len(fb_alerts) == 1


# ---------------------------------------------------------------------------
# Alert history accumulation
# ---------------------------------------------------------------------------


class TestAlertHistory:
    def test_history_accumulates(self, store, detector):
        for _ in range(20):
            _add_feedback(store, query="python hi", age_seconds=200000)
        for _ in range(10):
            _add_feedback(store, query="rust " + "z" * 300, age_seconds=10)

        alerts = detector.check()
        assert len(alerts) > 0
        assert detector.alert_count == len(alerts)
        assert detector.last_alert is alerts[-1]

        # A second run appends more history.
        before = detector.alert_count
        more = detector.check()
        assert detector.alert_count == before + len(more)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestExtractTopics:
    def test_extracts_keywords(self, detector):
        fb = [Feedback(query="My PYTHON and sql code")]
        topics = detector._extract_topics(fb)
        assert topics["python"] == 1
        assert topics["database"] == 1

    def test_empty_for_unknown(self, detector):
        topics = detector._extract_topics([Feedback(query="nothing here")])
        assert len(topics) == 0

    def test_sql_and_database_both_map(self, detector):
        # Both "database" and "sql" keywords map to topic "database".
        topics = detector._extract_topics([Feedback(query="database sql")])
        assert topics["database"] == 2


class TestBinDistribution:
    def test_normalizes_to_one(self, detector):
        bins = [0, 20, 50, 100, 200, 500, float("inf")]
        dist = detector._bin_distribution([10, 10, 30, 60], bins)
        assert pytest.approx(sum(dist)) == 1.0

    def test_empty_returns_uniform(self, detector):
        bins = [0, 10, 20]
        dist = detector._bin_distribution([], bins)
        # 2 buckets, uniform.
        assert dist == [0.5, 0.5]
        assert pytest.approx(sum(dist)) == 1.0

    def test_value_in_correct_bin(self, detector):
        bins = [0, 10, 20]
        dist = detector._bin_distribution([5, 5], bins)
        assert dist == [1.0, 0.0]


class TestKLDivergence:
    def test_identical_is_zero(self, detector):
        p = [0.25, 0.25, 0.25, 0.25]
        assert detector._kl_divergence(p, p) == pytest.approx(0.0)

    def test_divergent_is_positive(self, detector):
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        assert detector._kl_divergence(p, q) > 0

    def test_never_negative(self, detector):
        # Even with zeros (handled by eps), result clamped at >= 0.
        p = [0.0, 1.0]
        q = [1.0, 0.0]
        assert detector._kl_divergence(p, q) >= 0.0
