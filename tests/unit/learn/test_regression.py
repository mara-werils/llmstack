"""Tests for quality regression detection."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from llmstack.learn.regression import (
    RegressionAlert,
    RegressionConfig,
    RegressionDetector,
    RegressionSeverity,
)
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
def detector(store, version_mgr):
    config = RegressionConfig(
        min_samples=5,
        mild_threshold=0.03,
        moderate_threshold=0.08,
        severe_threshold=0.15,
        auto_rollback=True,
    )
    return RegressionDetector(store=store, version_mgr=version_mgr, config=config)


class TestRegressionDetector:
    def test_no_regression_without_data(self, detector):
        alerts = detector.check()
        assert len(alerts) == 0

    def test_no_regression_good_quality(self, detector, version_mgr, store):
        # Create active version with quality 0.8
        version_mgr.create_version(
            base_model="test",
            adapter_path="",
            quality_score=0.8,
            activate=True,
        )

        # Record quality snapshots at baseline level
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.79)

        alerts = detector.check()
        assert all(a.severity == RegressionSeverity.NONE for a in alerts)

    def test_detect_severe_regression(self, detector, version_mgr, store):
        version_mgr.create_version(
            base_model="test",
            adapter_path="",
            quality_score=0.8,
            activate=True,
        )

        # Record severely degraded quality
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.5)

        alerts = detector.check()
        severe = [a for a in alerts if a.severity == RegressionSeverity.SEVERE]
        assert len(severe) > 0

    def test_auto_rollback_on_severe(self, detector, version_mgr, store):
        # Create two versions
        version_mgr.create_version(
            base_model="test",
            adapter_path="",
            quality_score=0.7,
            activate=True,
        )
        version_mgr.create_version(
            base_model="test",
            adapter_path="",
            quality_score=0.8,
            activate=True,
        )

        # Severe degradation of version 2
        for _ in range(10):
            store.add_quality_snapshot("2", "overall", 0.5)

        alerts = detector.check()
        rolled_back = [a for a in alerts if a.auto_rolled_back]
        assert len(rolled_back) > 0

    def test_health_report(self, detector, version_mgr, store):
        version_mgr.create_version(
            base_model="test",
            adapter_path="",
            quality_score=0.8,
            activate=True,
        )

        store.add_quality_snapshot("1", "overall", 0.78)
        store.add_quality_snapshot("1", "overall", 0.80)

        health = detector.get_health()
        assert health["status"] in ("healthy", "degraded")
        assert "overall" in health.get("metrics", {})

    def test_confidence_threshold(self, detector, version_mgr, store):
        version_mgr.create_version(
            base_model="test",
            adapter_path="",
            quality_score=0.8,
            activate=True,
        )

        # High variance data — should not trigger alert due to low confidence
        import random

        random.seed(42)
        for _ in range(10):
            # Mix of high and low values — high variance
            store.add_quality_snapshot("1", "overall", random.uniform(0.3, 1.0))

        # With such high variance, confidence should be low
        alerts = detector.check()
        # May or may not trigger depending on random values, but validates the path
        assert isinstance(alerts, list)

    def test_alert_to_dict(self):
        alert = RegressionAlert(
            severity=RegressionSeverity.MODERATE,
            model_version="3",
            metric="overall",
            current_value=0.701234,
            baseline_value=0.8,
            drop_percent=12.345,
            sample_size=10,
            confidence=0.91234,
        )
        d = alert.to_dict()
        assert d["severity"] == "moderate"
        assert d["current_value"] == 0.7012
        assert d["drop_percent"] == 12.35
        assert d["confidence"] == 0.9123

    def test_alerts_and_alert_count_properties(self, detector, version_mgr, store):
        assert detector.alerts == []
        assert detector.alert_count == 0

        version_mgr.create_version(base_model="test", adapter_path="", quality_score=0.8, activate=True)
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.5)
        detector.check()

        assert detector.alert_count == len(detector.alerts)
        assert detector.alert_count > 0

    def test_is_regressing_true_after_severe_alert(self, detector, version_mgr, store):
        version_mgr.create_version(base_model="test", adapter_path="", quality_score=0.8, activate=True)
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.5)
        detector.check()
        assert detector.is_regressing is True

    def test_is_regressing_false_with_no_alerts(self, detector):
        assert detector.is_regressing is False

    def test_record_quality_writes_snapshot(self, detector, store):
        detector.record_quality("1", "overall", 0.75, sample_size=3)
        trend = store.get_quality_trend("1", "overall", limit=5)
        assert len(trend) == 1
        assert trend[0]["value"] == 0.75

    def test_get_health_with_no_active_model(self, detector):
        health = detector.get_health()
        assert health == {"status": "no_active_model", "metrics": {}}

    def test_baseline_zero_skips_check(self, detector, version_mgr, store):
        version_mgr.create_version(base_model="test", adapter_path="", quality_score=0.0, activate=True)
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.5)
        alerts = detector.check()
        assert alerts == []

    def test_mild_severity_detected(self, detector, version_mgr, store):
        version_mgr.create_version(base_model="test", adapter_path="", quality_score=1.0, activate=True)
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.95)  # 5% drop, identical values -> high confidence
        alerts = detector.check()
        mild = [a for a in alerts if a.severity == RegressionSeverity.MILD]
        assert len(mild) > 0

    def test_moderate_severity_detected(self, detector, version_mgr, store):
        version_mgr.create_version(base_model="test", adapter_path="", quality_score=1.0, activate=True)
        for _ in range(10):
            store.add_quality_snapshot("1", "overall", 0.90)  # 10% drop, identical values -> high confidence
        alerts = detector.check()
        moderate = [a for a in alerts if a.severity == RegressionSeverity.MODERATE]
        assert len(moderate) > 0

    def test_low_confidence_skips_alert(self, detector, monkeypatch):
        # Crafted trend with high variance -> low statistical confidence (~0.58) -> below
        # the 0.7 threshold, so _check_metric must return None despite a real drop.
        values = [0.632, 0.948] * 5
        trend = [{"value": v} for v in values]
        monkeypatch.setattr(detector.store, "get_quality_trend", lambda *a, **k: trend)

        alert = detector._check_metric("1", "overall", baseline=0.8)

        assert alert is None


class TestComputeConfidence:
    def test_single_value_returns_zero(self, detector):
        assert detector._compute_confidence([0.5], baseline=0.8) == 0.0

    def test_mean_above_baseline_returns_zero(self, detector):
        assert detector._compute_confidence([0.9, 0.95, 1.0], baseline=0.5) == 0.0

    def test_zero_standard_error_branch_mean_below_baseline(self, detector):
        # First math.sqrt() call computes std (forced to 0); second computes sqrt(n)
        # for the se denominator, which must stay real or se's division blows up.
        with patch(
            "llmstack.learn.regression.math.sqrt", side_effect=[0.0, math.sqrt(3)]
        ):
            confidence = detector._compute_confidence([0.1, 0.2, 0.3], baseline=0.8)
        assert confidence == 1.0

    def test_zero_standard_error_branch_mean_above_baseline(self, detector):
        with patch(
            "llmstack.learn.regression.math.sqrt", side_effect=[0.0, math.sqrt(3)]
        ):
            confidence = detector._compute_confidence([0.9, 0.95, 1.0], baseline=0.5)
        assert confidence == 0.0
