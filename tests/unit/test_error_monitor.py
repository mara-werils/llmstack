"""Tests for error rate monitoring."""

from __future__ import annotations

import pytest

from llmstack.observe.error_monitor import (
    ErrorMonitorConfig,
    ErrorRateMonitor,
)


@pytest.fixture
def monitor():
    return ErrorRateMonitor()


class TestErrorRateMonitor:
    def test_no_errors_initially(self, monitor):
        summary = monitor.get_summary()
        assert summary["total_errors"] == 0

    def test_record_error(self, monitor):
        alerts = monitor.record_error("timeout", provider="openai")
        summary = monitor.get_summary()
        assert summary["total_errors"] == 1

    def test_consecutive_errors_alert(self):
        config = ErrorMonitorConfig(consecutive_threshold=3)
        mon = ErrorRateMonitor(config)
        for i in range(3):
            alerts = mon.record_error("timeout", provider="openai")
        assert any(a.alert_type == "consecutive_errors" for a in alerts)

    def test_success_resets_consecutive(self):
        config = ErrorMonitorConfig(consecutive_threshold=3)
        mon = ErrorRateMonitor(config)
        mon.record_error("timeout", provider="openai")
        mon.record_error("timeout", provider="openai")
        mon.record_success(provider="openai")
        alerts = mon.record_error("timeout", provider="openai")
        assert not any(a.alert_type == "consecutive_errors" for a in alerts)

    def test_error_rate_warning(self):
        config = ErrorMonitorConfig(warning_threshold=0.3, critical_threshold=0.8)
        mon = ErrorRateMonitor(config)
        # 1 success, then errors
        mon.record_success(provider="api")
        for _ in range(5):
            alerts = mon.record_error("500", provider="api")
        # Should have warning or critical
        assert len(alerts) > 0

    def test_error_rate_below_threshold_no_alert(self):
        config = ErrorMonitorConfig(
            warning_threshold=0.5,
            consecutive_threshold=100,
        )
        mon = ErrorRateMonitor(config)
        # Many successes, few errors
        for _ in range(100):
            mon.record_success(provider="api")
        alerts = mon.record_error("500", provider="api")
        # Rate is very low, should not trigger rate alert
        rate_alerts = [a for a in alerts if a.alert_type == "error_rate"]
        assert len(rate_alerts) == 0

    def test_summary_structure(self, monitor):
        monitor.record_error("timeout", provider="openai")
        monitor.record_error("500", provider="anthropic")
        summary = monitor.get_summary()
        assert "by_type" in summary
        assert "by_provider" in summary
        assert "timeout" in summary["by_type"]

    def test_error_event_types(self, monitor):
        monitor.record_error("timeout", provider="p1")
        monitor.record_error("rate_limit", provider="p1")
        summary = monitor.get_summary()
        assert summary["by_type"]["timeout"] == 1
        assert summary["by_type"]["rate_limit"] == 1

    def test_alert_serialization(self):
        config = ErrorMonitorConfig(consecutive_threshold=1)
        mon = ErrorRateMonitor(config)
        alerts = mon.record_error("timeout", provider="openai")
        if alerts:
            d = alerts[0].to_dict()
            assert "alert_type" in d
            assert "severity" in d
            assert "message" in d

    def test_record_request(self, monitor):
        monitor.record_request(provider="openai", endpoint="/v1/chat")
        summary = monitor.get_summary()
        assert summary["total_errors"] == 0
