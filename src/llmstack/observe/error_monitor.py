"""Error rate monitoring with automatic alerting.

Tracks error rates across providers and endpoints, firing alerts
when error rates exceed configured thresholds.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """A recorded error event."""

    timestamp: float
    error_type: str
    provider: str = ""
    endpoint: str = ""
    message: str = ""


@dataclass
class ErrorAlert:
    """An alert triggered by error rate threshold."""

    alert_type: str  # "error_rate", "error_spike", "consecutive_errors"
    severity: str  # "warning", "critical"
    message: str
    provider: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "provider": self.provider,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class ErrorMonitorConfig:
    """Configuration for error monitoring."""

    # Error rate threshold for warning (0.0-1.0)
    warning_threshold: float = 0.05

    # Error rate threshold for critical (0.0-1.0)
    critical_threshold: float = 0.15

    # Time window for rate calculation (seconds)
    window_seconds: float = 300.0  # 5 minutes

    # Consecutive errors before alert
    consecutive_threshold: int = 5

    # Maximum events to keep
    max_events: int = 5000


class ErrorRateMonitor:
    """Monitors error rates and fires alerts on threshold violations.

    Tracks errors by provider and endpoint, computing rolling error
    rates and detecting error spikes.
    """

    def __init__(self, config: ErrorMonitorConfig | None = None):
        self.config = config or ErrorMonitorConfig()
        self._events: list[ErrorEvent] = []
        self._request_counts: Counter = Counter()
        self._consecutive: dict[str, int] = {}
        self._lock = threading.Lock()
        self._alerts: list[ErrorAlert] = []

    def record_request(self, provider: str = "", endpoint: str = "") -> None:
        """Record a request (success or error)."""
        with self._lock:
            key = f"{provider}:{endpoint}"
            self._request_counts[key] += 1

    def record_error(
        self,
        error_type: str,
        provider: str = "",
        endpoint: str = "",
        message: str = "",
    ) -> list[ErrorAlert]:
        """Record an error event and return any triggered alerts."""
        with self._lock:
            event = ErrorEvent(
                timestamp=time.time(),
                error_type=error_type,
                provider=provider,
                endpoint=endpoint,
                message=message,
            )
            self._events.append(event)
            if len(self._events) > self.config.max_events:
                self._events = self._events[-self.config.max_events :]

            # Track consecutive errors
            key = provider or "global"
            self._consecutive[key] = self._consecutive.get(key, 0) + 1

            return self._check_alerts(provider, endpoint)

    def record_success(self, provider: str = "", endpoint: str = "") -> None:
        """Record a successful request (resets consecutive error count)."""
        with self._lock:
            key = provider or "global"
            self._consecutive[key] = 0
            req_key = f"{provider}:{endpoint}"
            self._request_counts[req_key] += 1

    def get_error_rate(self, provider: str = "") -> float:
        """Get the current error rate for a provider."""
        with self._lock:
            cutoff = time.time() - self.config.window_seconds
            errors = [
                e
                for e in self._events
                if e.timestamp >= cutoff and (not provider or e.provider == provider)
            ]
            total_key = f"{provider}:"
            total = sum(
                v
                for k, v in self._request_counts.items()
                if k.startswith(total_key) or not provider
            )
            if total == 0:
                return 0.0
            return len(errors) / total

    def get_summary(self) -> dict[str, Any]:
        """Get error monitoring summary."""
        with self._lock:
            cutoff = time.time() - self.config.window_seconds
            recent = [e for e in self._events if e.timestamp >= cutoff]
            by_type = Counter(e.error_type for e in recent)
            by_provider = Counter(e.provider for e in recent if e.provider)

            return {
                "total_errors": len(self._events),
                "recent_errors": len(recent),
                "window_seconds": self.config.window_seconds,
                "by_type": dict(by_type),
                "by_provider": dict(by_provider),
                "active_alerts": len(self._alerts),
                "alerts": [a.to_dict() for a in self._alerts[-10:]],
            }

    def _check_alerts(self, provider: str, endpoint: str) -> list[ErrorAlert]:
        """Check if any alert thresholds are exceeded."""
        alerts: list[ErrorAlert] = []

        # Check consecutive errors
        key = provider or "global"
        if self._consecutive.get(key, 0) >= self.config.consecutive_threshold:
            alert = ErrorAlert(
                alert_type="consecutive_errors",
                severity="critical",
                message=f"Consecutive errors ({self._consecutive[key]}) for {key}",
                provider=provider,
                details={"count": self._consecutive[key]},
            )
            alerts.append(alert)

        # Check error rate
        cutoff = time.time() - self.config.window_seconds
        recent_errors = [
            e
            for e in self._events
            if e.timestamp >= cutoff and (not provider or e.provider == provider)
        ]
        total_key = f"{provider}:{endpoint}"
        total = self._request_counts.get(total_key, 0) + len(recent_errors)
        if total > 0:
            rate = len(recent_errors) / total
            if rate >= self.config.critical_threshold:
                alert = ErrorAlert(
                    alert_type="error_rate",
                    severity="critical",
                    message=f"Error rate {rate:.1%} exceeds critical threshold",
                    provider=provider,
                    details={"rate": round(rate, 4), "errors": len(recent_errors)},
                )
                alerts.append(alert)
            elif rate >= self.config.warning_threshold:
                alert = ErrorAlert(
                    alert_type="error_rate",
                    severity="warning",
                    message=f"Error rate {rate:.1%} exceeds warning threshold",
                    provider=provider,
                    details={"rate": round(rate, 4), "errors": len(recent_errors)},
                )
                alerts.append(alert)

        self._alerts.extend(alerts)
        return alerts
