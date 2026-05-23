"""Provider health checker with periodic probing.

Monitors the health of downstream LLM providers and tracks their
availability, latency, and error rates over time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Provider health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderHealthRecord:
    """Health record for a single provider."""

    provider: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    error_message: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return (self.total_checks - self.total_failures) / self.total_checks

    @property
    def uptime_pct(self) -> float:
        return round(self.success_rate * 100, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status.value,
            "last_check": self.last_check,
            "latency_ms": round(self.latency_ms, 2),
            "consecutive_failures": self.consecutive_failures,
            "success_rate": round(self.success_rate, 4),
            "uptime_pct": self.uptime_pct,
            "total_checks": self.total_checks,
            "error_message": self.error_message,
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checking."""

    # Consecutive failures before marking unhealthy
    failure_threshold: int = 3

    # Consecutive failures for degraded status
    degraded_threshold: int = 1

    # Check interval (seconds)
    check_interval: float = 60.0

    # Request timeout for health checks (seconds)
    timeout: float = 10.0

    # Latency threshold for degraded status (ms)
    latency_threshold_ms: float = 5000.0


class ProviderHealthChecker:
    """Tracks and reports on provider health.

    Records success/failure of provider interactions and computes
    health status based on consecutive failures and latency.
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        self.config = config or HealthCheckConfig()
        self._records: dict[str, ProviderHealthRecord] = {}

    def record_success(self, provider: str, latency_ms: float = 0.0) -> None:
        """Record a successful provider interaction."""
        record = self._get_record(provider)
        record.total_checks += 1
        record.last_check = time.time()
        record.last_success = time.time()
        record.latency_ms = latency_ms
        record.consecutive_failures = 0
        record.error_message = ""

        if latency_ms > self.config.latency_threshold_ms:
            record.status = HealthStatus.DEGRADED
        else:
            record.status = HealthStatus.HEALTHY

    def record_failure(self, provider: str, error: str = "") -> None:
        """Record a failed provider interaction."""
        record = self._get_record(provider)
        record.total_checks += 1
        record.total_failures += 1
        record.last_check = time.time()
        record.last_failure = time.time()
        record.consecutive_failures += 1
        record.error_message = error

        if record.consecutive_failures >= self.config.failure_threshold:
            record.status = HealthStatus.UNHEALTHY
        elif record.consecutive_failures >= self.config.degraded_threshold:
            record.status = HealthStatus.DEGRADED

    def get_status(self, provider: str) -> HealthStatus:
        """Get the current health status of a provider."""
        record = self._records.get(provider)
        return record.status if record else HealthStatus.UNKNOWN

    def get_health(self, provider: str | None = None) -> dict[str, Any]:
        """Get health report for one or all providers."""
        if provider:
            record = self._records.get(provider)
            if record:
                return record.to_dict()
            return {"provider": provider, "status": "unknown"}

        healthy = sum(1 for r in self._records.values() if r.status == HealthStatus.HEALTHY)
        total = len(self._records)

        return {
            "healthy_count": healthy,
            "total_providers": total,
            "providers": {
                name: rec.to_dict() for name, rec in self._records.items()
            },
        }

    def get_healthy_providers(self) -> list[str]:
        """Get list of healthy provider names."""
        return [
            name for name, rec in self._records.items()
            if rec.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        ]

    def _get_record(self, provider: str) -> ProviderHealthRecord:
        if provider not in self._records:
            self._records[provider] = ProviderHealthRecord(provider=provider)
        return self._records[provider]
