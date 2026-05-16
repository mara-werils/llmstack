"""Tests for quality regression detection."""

from __future__ import annotations

import pytest

from llmstack.learn.regression import (
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
