"""Tests for latency percentile tracking."""

from __future__ import annotations

import pytest

from llmstack.observe.latency import LatencyConfig, LatencyTracker


@pytest.fixture
def tracker():
    return LatencyTracker()


class TestLatencyTracker:
    def test_no_data_returns_zero(self, tracker):
        assert tracker.percentile(50) == 0.0
        assert tracker.percentile(99) == 0.0

    def test_single_value(self, tracker):
        tracker.record(100.0)
        assert tracker.percentile(50) == 100.0
        assert tracker.percentile(99) == 100.0

    def test_p50_is_median(self, tracker):
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            tracker.record(float(v))
        p50 = tracker.percentile(50)
        assert 40 <= p50 <= 60

    def test_p99_is_high(self, tracker):
        for v in range(1, 101):
            tracker.record(float(v))
        p99 = tracker.percentile(99)
        assert p99 >= 95

    def test_get_percentiles(self, tracker):
        for v in range(1, 101):
            tracker.record(float(v))
        pcts = tracker.get_percentiles()
        assert "p50" in pcts
        assert "p95" in pcts
        assert "p99" in pcts
        assert pcts["p50"] < pcts["p95"] < pcts["p99"]

    def test_get_summary(self, tracker):
        for v in [10, 20, 30]:
            tracker.record(float(v))
        summary = tracker.get_summary()
        assert summary["count"] == 3
        assert summary["mean"] == 20.0
        assert summary["min"] == 10.0
        assert summary["max"] == 30.0

    def test_empty_summary(self, tracker):
        summary = tracker.get_summary()
        assert summary["count"] == 0

    def test_reset(self, tracker):
        tracker.record(100.0)
        tracker.reset()
        assert tracker.percentile(50) == 0.0

    def test_max_samples(self):
        config = LatencyConfig(max_samples=10)
        tracker = LatencyTracker(config)
        for i in range(20):
            tracker.record(float(i))
        summary = tracker.get_summary()
        assert summary["count"] <= 10

    def test_window_filtering(self):
        config = LatencyConfig(window_seconds=0.01)
        tracker = LatencyTracker(config)
        tracker.record(100.0)
        import time

        time.sleep(0.02)
        # Old samples should be excluded
        assert tracker.percentile(50) == 0.0
