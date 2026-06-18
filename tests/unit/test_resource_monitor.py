"""Tests for system resource monitor."""

from __future__ import annotations

import pytest

from llmstack.core.resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    ResourceThresholds,
)


@pytest.fixture
def monitor():
    return ResourceMonitor()


class TestResourceMonitor:
    def test_snapshot(self, monitor):
        snap = monitor.snapshot()
        assert snap.memory_total_mb > 0
        assert snap.disk_total_gb > 0
        assert 0 <= snap.memory_percent <= 100
        assert 0 <= snap.disk_percent <= 100

    def test_snapshot_to_dict(self, monitor):
        snap = monitor.snapshot()
        d = snap.to_dict()
        assert "memory" in d
        assert "disk" in d
        assert "cpu_percent" in d

    def test_check_health_returns_status(self, monitor):
        health = monitor.check_health()
        assert "status" in health
        assert health["status"] in ("healthy", "warning", "critical")
        assert "warnings" in health
        assert "snapshot" in health

    def test_check_health_with_strict_thresholds(self):
        # Very low thresholds to trigger warnings
        thresholds = ResourceThresholds(
            memory_warning_pct=1.0,
            disk_warning_pct=1.0,
        )
        mon = ResourceMonitor(thresholds)
        health = mon.check_health()
        assert health["status"] in ("warning", "critical")
        assert len(health["warnings"]) > 0

    def test_history_maintained(self, monitor):
        monitor.snapshot()
        monitor.snapshot()
        monitor.snapshot()
        trend = monitor.get_trend(minutes=60)
        assert trend["samples"] == 3

    def test_trend_empty(self, monitor):
        trend = monitor.get_trend()
        assert trend["samples"] == 0

    def test_trend_with_data(self, monitor):
        for _ in range(5):
            monitor.snapshot()
        trend = monitor.get_trend(minutes=60)
        assert trend["samples"] == 5
        assert "cpu_avg" in trend
        assert "memory_avg_pct" in trend

    def test_max_history(self):
        mon = ResourceMonitor()
        mon._max_history = 5
        for _ in range(10):
            mon.snapshot()
        assert len(mon._history) == 5

    def test_check_health_cpu_warning_only(self, monkeypatch):
        # High CPU but normal memory/disk usage -> warning from CPU branch alone.
        monkeypatch.setattr("psutil.cpu_percent", lambda interval=0: 99.0)
        mon = ResourceMonitor(ResourceThresholds(cpu_warning_pct=50.0))
        health = mon.check_health()
        assert health["status"] == "warning"
        assert any("CPU" in w for w in health["warnings"])

    def test_check_health_with_critical_thresholds(self):
        thresholds = ResourceThresholds(
            memory_warning_pct=0.5,
            memory_critical_pct=0.5,
            disk_warning_pct=0.5,
            disk_critical_pct=0.5,
        )
        mon = ResourceMonitor(thresholds)
        health = mon.check_health()
        assert health["status"] == "critical"
        assert any("CRITICAL" in w for w in health["warnings"])


class TestResourceSnapshot:
    def test_memory_free_mb(self):
        snap = ResourceSnapshot(
            timestamp=0,
            cpu_percent=0,
            memory_total_mb=1000,
            memory_used_mb=400,
            memory_percent=40.0,
            disk_total_gb=100,
            disk_used_gb=30,
            disk_percent=30.0,
        )
        assert snap.memory_free_mb == 600

    def test_disk_free_gb(self):
        snap = ResourceSnapshot(
            timestamp=0,
            cpu_percent=0,
            memory_total_mb=1000,
            memory_used_mb=400,
            memory_percent=40.0,
            disk_total_gb=100,
            disk_used_gb=30,
            disk_percent=30.0,
        )
        assert snap.disk_free_gb == 70
