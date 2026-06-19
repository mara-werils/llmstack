"""Tests for learning analytics — metrics, timelines, status, and recommendations."""

from __future__ import annotations

import time

import pytest

from llmstack.learn.analytics import (
    LearningAnalytics,
    LearningMetrics,
    TimeSeriesPoint,
)
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore
from llmstack.learn.versions import ModelVersionManager


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def version_mgr(store, tmp_path):
    return ModelVersionManager(store=store, versions_dir=tmp_path / "versions")


@pytest.fixture
def analytics(store, version_mgr):
    return LearningAnalytics(store=store, version_mgr=version_mgr)


def _add(store, fb_type, query="A reasonably long query here", response="A response", **kw):
    store.add_feedback(Feedback(feedback_type=fb_type, query=query, response=response, **kw))


# --------------------------------------------------------------------------- #
# LearningMetrics dataclass
# --------------------------------------------------------------------------- #
class TestLearningMetrics:
    def test_defaults(self):
        m = LearningMetrics()
        assert m.total_feedback == 0
        assert m.quality_trend == "stable"

    def test_to_dict_structure_and_rounding(self):
        m = LearningMetrics(
            total_feedback=10,
            positive_rate=0.123456,
            correction_rate=0.654321,
            feedback_per_day=3.14159,
            unused_feedback=4,
            total_train_runs=2,
            total_versions=3,
            avg_dataset_size=123.4,
            avg_train_time=55.57,
            current_quality=0.987654,
            quality_improvement=0.111111,
            best_quality=0.999999,
            quality_trend="improving",
            feedback_to_improvement_ratio=0.012345,
        )
        d = m.to_dict()
        assert d["feedback"]["total"] == 10
        assert d["feedback"]["positive_rate"] == 0.123
        assert d["feedback"]["correction_rate"] == 0.654
        assert d["feedback"]["per_day"] == 3.1
        assert d["feedback"]["unused"] == 4
        assert d["training"]["total_runs"] == 2
        assert d["training"]["total_versions"] == 3
        assert d["training"]["avg_dataset_size"] == 123.0
        assert d["training"]["avg_train_time_sec"] == 55.6
        assert d["quality"]["current"] == 0.9877
        assert d["quality"]["improvement"] == 0.1111
        assert d["quality"]["best"] == 1.0
        assert d["quality"]["trend"] == "improving"
        assert d["efficiency"]["feedback_to_improvement"] == 0.0123


class TestTimeSeriesPoint:
    def test_defaults(self):
        p = TimeSeriesPoint(timestamp=1.0, value=2.0)
        assert p.label == ""
        assert p.timestamp == 1.0
        assert p.value == 2.0


# --------------------------------------------------------------------------- #
# compute_metrics — feedback metrics
# --------------------------------------------------------------------------- #
class TestComputeMetricsFeedback:
    def test_empty_store(self, analytics):
        m = analytics.compute_metrics()
        assert m.total_feedback == 0
        assert m.unused_feedback == 0
        assert m.positive_rate == 0.0
        assert m.correction_rate == 0.0
        assert m.feedback_per_day == 0.0

    def test_positive_rate(self, analytics, store):
        # 3 thumbs up, 1 thumbs down => positive_rate 0.75
        for _ in range(3):
            _add(store, FeedbackType.THUMBS_UP)
        _add(store, FeedbackType.THUMBS_DOWN)

        m = analytics.compute_metrics()
        assert m.total_feedback == 4
        assert m.positive_rate == pytest.approx(0.75)

    def test_correction_rate(self, analytics, store):
        # 2 corrections + 1 edit out of 5 total => 0.6
        for _ in range(2):
            _add(store, FeedbackType.CORRECTION, correction="a corrected answer")
        _add(store, FeedbackType.EDIT, edit_diff="some diff")
        _add(store, FeedbackType.THUMBS_UP)
        _add(store, FeedbackType.THUMBS_DOWN)

        m = analytics.compute_metrics()
        assert m.total_feedback == 5
        assert m.correction_rate == pytest.approx(0.6)

    def test_no_rated_feedback_keeps_zero_positive_rate(self, analytics, store):
        # Only corrections — no thumbs => positive_rate stays 0.0 (total_rated == 0)
        _add(store, FeedbackType.CORRECTION, correction="a corrected answer")
        m = analytics.compute_metrics()
        assert m.positive_rate == 0.0
        assert m.total_feedback == 1

    def test_feedback_per_day_spread_over_days(self, analytics, store):
        # get_feedback(limit=1) returns the most-recent row (ORDER BY ts DESC).
        # Here the only/most-recent feedback is ~2 days old, so days >= 2 and
        # feedback_per_day < total_feedback.
        old_ts = time.time() - 2 * 86400
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query="An older interaction query",
                response="old response",
                timestamp=old_ts,
            )
        )
        m = analytics.compute_metrics()
        assert m.feedback_per_day > 0
        # 1 feedback over ~2 days -> ~0.5/day, below total of 1
        assert m.feedback_per_day < m.total_feedback

    def test_feedback_per_day_recent_clamped_to_one_day(self, analytics, store):
        # Very recent feedback -> days clamped to 1 -> per_day == total_feedback
        _add(store, FeedbackType.THUMBS_UP)
        _add(store, FeedbackType.THUMBS_UP)
        m = analytics.compute_metrics()
        assert m.feedback_per_day == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# compute_metrics — training & quality metrics
# --------------------------------------------------------------------------- #
class TestComputeMetricsQuality:
    def test_training_counts(self, analytics, store):
        store.add_train_run("1", "base", feedback_count=10, dataset_size=20)
        store.add_model_version("1", "base", quality_score=0.5, is_active=True)
        m = analytics.compute_metrics()
        assert m.total_train_runs == 1
        assert m.total_versions == 1

    def test_single_version_quality(self, analytics, version_mgr):
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.6, activate=True
        )
        m = analytics.compute_metrics()
        assert m.current_quality == pytest.approx(0.6)
        assert m.best_quality == pytest.approx(0.6)
        # Only one version -> no improvement computed
        assert m.quality_improvement == 0.0
        assert m.quality_trend == "stable"

    def test_no_active_version_current_quality_zero(self, analytics, version_mgr):
        # Create version but don't activate it -> active is None branch
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.7, activate=False
        )
        m = analytics.compute_metrics()
        assert m.current_quality == 0.0
        assert m.best_quality == pytest.approx(0.7)

    def test_quality_improving_trend(self, analytics, version_mgr):
        # Create ascending-quality versions; latest (v3) is best & active
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.5, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.7, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.9, activate=True
        )
        m = analytics.compute_metrics()
        # latest 0.9 - first 0.5 = 0.4
        assert m.quality_improvement == pytest.approx(0.4)
        assert m.quality_trend == "improving"
        assert m.best_quality == pytest.approx(0.9)
        assert m.current_quality == pytest.approx(0.9)

    def test_quality_declining_trend(self, analytics, version_mgr):
        # Descending quality across versions -> declining
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.9, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.7, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.5, activate=True
        )
        m = analytics.compute_metrics()
        # latest 0.5 - first 0.9 = -0.4
        assert m.quality_improvement == pytest.approx(-0.4)
        assert m.quality_trend == "declining"

    def test_quality_stable_trend(self, analytics, version_mgr):
        # Nearly identical scores -> within 0.01 -> stays stable
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.80, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.805, activate=True
        )
        m = analytics.compute_metrics()
        assert m.quality_trend == "stable"

    def test_feedback_to_improvement_ratio(self, analytics, store, version_mgr):
        # Need positive quality_improvement and feedback > 0
        for _ in range(10):
            _add(store, FeedbackType.THUMBS_UP)
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.4, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.8, activate=True
        )
        m = analytics.compute_metrics()
        assert m.quality_improvement == pytest.approx(0.4)
        assert m.feedback_to_improvement_ratio == pytest.approx(0.4 / 10)

    def test_no_ratio_when_no_improvement(self, analytics, store, version_mgr):
        # Declining improvement -> ratio stays 0
        for _ in range(10):
            _add(store, FeedbackType.THUMBS_UP)
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.8, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.4, activate=True
        )
        m = analytics.compute_metrics()
        assert m.feedback_to_improvement_ratio == 0.0


# --------------------------------------------------------------------------- #
# get_quality_timeline
# --------------------------------------------------------------------------- #
class TestQualityTimeline:
    def test_empty(self, analytics):
        assert analytics.get_quality_timeline() == []

    def test_filters_zero_quality_and_orders_chronologically(self, analytics, version_mgr):
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.0, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.5, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.7, activate=True
        )
        timeline = analytics.get_quality_timeline()
        # The zero-quality version is filtered out
        assert len(timeline) == 2
        assert all(isinstance(p, TimeSeriesPoint) for p in timeline)
        # reversed(list_versions) -> oldest non-zero first (v2=0.5, then v3=0.7)
        assert [p.value for p in timeline] == [0.5, 0.7]
        assert timeline[0].label == "v2"
        assert timeline[1].label == "v3"

    def test_respects_limit(self, analytics, version_mgr):
        for q in (0.3, 0.4, 0.5):
            version_mgr.create_version(
                base_model="base", adapter_path="", quality_score=q, activate=False
            )
        timeline = analytics.get_quality_timeline(limit=2)
        assert len(timeline) == 2


# --------------------------------------------------------------------------- #
# get_feedback_timeline
# --------------------------------------------------------------------------- #
class TestFeedbackTimeline:
    def test_empty(self, analytics):
        assert analytics.get_feedback_timeline() == []

    def test_buckets_recent_feedback(self, analytics, store):
        for _ in range(3):
            _add(store, FeedbackType.THUMBS_UP)
        timeline = analytics.get_feedback_timeline(bucket_hours=24, limit=30)
        assert len(timeline) == 30
        # All recent feedback falls into bucket 0
        assert timeline[0].value == 3.0
        assert timeline[0].label == "-0h"
        assert timeline[1].value == 0.0
        assert all(isinstance(p, TimeSeriesPoint) for p in timeline)

    def test_old_feedback_outside_limit_excluded(self, analytics, store):
        # Feedback ~5 days old; with 1h buckets and limit 3, it lands far beyond
        old_ts = time.time() - 5 * 86400
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query="An old query here for testing",
                response="resp",
                timestamp=old_ts,
            )
        )
        _add(store, FeedbackType.THUMBS_UP)  # recent
        timeline = analytics.get_feedback_timeline(bucket_hours=1, limit=3)
        assert len(timeline) == 3
        # Only the recent one counted (old one out of range)
        assert sum(p.value for p in timeline) == 1.0
        assert timeline[0].value == 1.0

    def test_distinct_buckets(self, analytics, store):
        now = time.time()
        # one ~25h ago (bucket 1 with 24h buckets), one recent (bucket 0)
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query="A query from yesterday here",
                response="resp",
                timestamp=now - 25 * 3600,
            )
        )
        _add(store, FeedbackType.THUMBS_UP)
        timeline = analytics.get_feedback_timeline(bucket_hours=24, limit=5)
        assert timeline[0].value == 1.0
        assert timeline[1].value == 1.0
        assert timeline[1].label == "-24h"


# --------------------------------------------------------------------------- #
# get_summary
# --------------------------------------------------------------------------- #
class TestGetSummary:
    def test_summary_keys(self, analytics, store):
        _add(store, FeedbackType.THUMBS_UP)
        summary = analytics.get_summary()
        assert set(summary.keys()) == {"status", "metrics", "recommendations"}
        assert isinstance(summary["recommendations"], list)
        assert isinstance(summary["metrics"], dict)

    def test_summary_inactive_status(self, analytics):
        summary = analytics.get_summary()
        assert summary["status"] == "inactive"


# --------------------------------------------------------------------------- #
# _compute_status
# --------------------------------------------------------------------------- #
class TestComputeStatus:
    def test_inactive(self, analytics):
        assert analytics._compute_status(LearningMetrics()) == "inactive"

    def test_collecting(self, analytics):
        m = LearningMetrics(total_feedback=5, total_versions=0)
        assert analytics._compute_status(m) == "collecting"

    def test_improving(self, analytics):
        m = LearningMetrics(total_feedback=5, total_versions=2, quality_trend="improving")
        assert analytics._compute_status(m) == "improving"

    def test_degrading(self, analytics):
        m = LearningMetrics(total_feedback=5, total_versions=2, quality_trend="declining")
        assert analytics._compute_status(m) == "degrading"

    def test_active(self, analytics):
        m = LearningMetrics(total_feedback=5, total_versions=2, quality_trend="stable")
        assert analytics._compute_status(m) == "active"


# --------------------------------------------------------------------------- #
# _get_recommendations
# --------------------------------------------------------------------------- #
class TestRecommendations:
    def test_no_feedback_recommendation(self, analytics):
        recs = analytics._get_recommendations(LearningMetrics())
        assert any("Start collecting feedback" in r for r in recs)

    def test_ready_to_train_recommendation(self, analytics):
        m = LearningMetrics(total_feedback=30, total_versions=0, unused_feedback=30)
        recs = analytics._get_recommendations(m)
        assert any("30 unused feedback items" in r for r in recs)
        assert any("learn train" in r for r in recs)

    def test_below_threshold_collecting_recommendation(self, analytics):
        m = LearningMetrics(total_feedback=10, total_versions=0, unused_feedback=10)
        recs = analytics._get_recommendations(m)
        assert any("10/25 threshold" in r for r in recs)

    def test_low_satisfaction_recommendation(self, analytics):
        m = LearningMetrics(
            total_feedback=20,
            total_versions=1,
            positive_rate=0.3,
            quality_trend="stable",
        )
        recs = analytics._get_recommendations(m)
        assert any("Low satisfaction rate" in r for r in recs)

    def test_no_low_satisfaction_when_too_few_feedback(self, analytics):
        # positive_rate low but only 5 feedback -> skip the low-satisfaction rec
        m = LearningMetrics(
            total_feedback=5,
            total_versions=1,
            positive_rate=0.1,
        )
        recs = analytics._get_recommendations(m)
        assert not any("Low satisfaction rate" in r for r in recs)

    def test_declining_recommendation(self, analytics):
        m = LearningMetrics(
            total_feedback=20,
            total_versions=2,
            positive_rate=0.9,
            quality_trend="declining",
        )
        recs = analytics._get_recommendations(m)
        assert any("Quality is declining" in r for r in recs)

    def test_high_correction_rate_recommendation(self, analytics):
        m = LearningMetrics(
            total_feedback=10,
            total_versions=1,
            positive_rate=0.9,
            correction_rate=0.6,
            quality_trend="stable",
        )
        recs = analytics._get_recommendations(m)
        assert any("High correction rate" in r for r in recs)

    def test_empty_recommendations_for_healthy_pipeline(self, analytics):
        # Everything good: feedback collected, versions exist, high satisfaction,
        # low corrections, improving trend -> no recommendations triggered
        m = LearningMetrics(
            total_feedback=50,
            total_versions=3,
            unused_feedback=2,
            positive_rate=0.9,
            correction_rate=0.1,
            quality_trend="improving",
        )
        recs = analytics._get_recommendations(m)
        assert recs == []


# --------------------------------------------------------------------------- #
# End-to-end through real collaborators
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_full_summary_improving_pipeline(self, analytics, store, version_mgr):
        for _ in range(30):
            _add(store, FeedbackType.THUMBS_UP)
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.5, activate=False
        )
        version_mgr.create_version(
            base_model="base", adapter_path="", quality_score=0.9, activate=True
        )
        summary = analytics.get_summary()
        assert summary["status"] == "improving"
        assert summary["metrics"]["quality"]["trend"] == "improving"
        assert summary["metrics"]["feedback"]["total"] == 30
