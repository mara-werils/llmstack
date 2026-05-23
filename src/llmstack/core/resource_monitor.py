"""System resource monitor for memory and disk tracking.

Monitors system resources (CPU, memory, disk) to provide health data
and early warnings when resources are running low.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Point-in-time snapshot of system resources."""

    timestamp: float
    cpu_percent: float
    memory_total_mb: int
    memory_used_mb: int
    memory_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory": {
                "total_mb": self.memory_total_mb,
                "used_mb": self.memory_used_mb,
                "percent": round(self.memory_percent, 1),
            },
            "disk": {
                "total_gb": round(self.disk_total_gb, 1),
                "used_gb": round(self.disk_used_gb, 1),
                "percent": round(self.disk_percent, 1),
            },
        }


@dataclass
class ResourceThresholds:
    """Thresholds for resource warnings."""

    memory_warning_pct: float = 80.0
    memory_critical_pct: float = 95.0
    disk_warning_pct: float = 85.0
    disk_critical_pct: float = 95.0
    cpu_warning_pct: float = 90.0


class ResourceMonitor:
    """Monitors system resources and provides health assessments."""

    def __init__(self, thresholds: ResourceThresholds | None = None):
        self.thresholds = thresholds or ResourceThresholds()
        self._history: list[ResourceSnapshot] = []
        self._max_history: int = 100

    def snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0),
            memory_total_mb=int(mem.total / (1024 * 1024)),
            memory_used_mb=int(mem.used / (1024 * 1024)),
            memory_percent=mem.percent,
            disk_total_gb=disk.total / (1024 ** 3),
            disk_used_gb=disk.used / (1024 ** 3),
            disk_percent=disk.percent,
        )

        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snap

    def check_health(self) -> dict[str, Any]:
        """Check resource health and return warnings."""
        snap = self.snapshot()
        warnings: list[str] = []
        status = "healthy"

        # Memory checks
        if snap.memory_percent >= self.thresholds.memory_critical_pct:
            warnings.append(f"CRITICAL: Memory usage at {snap.memory_percent:.1f}%")
            status = "critical"
        elif snap.memory_percent >= self.thresholds.memory_warning_pct:
            warnings.append(f"WARNING: Memory usage at {snap.memory_percent:.1f}%")
            status = "warning"

        # Disk checks
        if snap.disk_percent >= self.thresholds.disk_critical_pct:
            warnings.append(f"CRITICAL: Disk usage at {snap.disk_percent:.1f}%")
            status = "critical"
        elif snap.disk_percent >= self.thresholds.disk_warning_pct:
            warnings.append(f"WARNING: Disk usage at {snap.disk_percent:.1f}%")
            if status != "critical":
                status = "warning"

        # CPU checks
        if snap.cpu_percent >= self.thresholds.cpu_warning_pct:
            warnings.append(f"WARNING: CPU usage at {snap.cpu_percent:.1f}%")
            if status != "critical":
                status = "warning"

        return {
            "status": status,
            "warnings": warnings,
            "snapshot": snap.to_dict(),
        }

    def get_trend(self, minutes: int = 60) -> dict[str, Any]:
        """Get resource usage trend over the specified period."""
        cutoff = time.time() - (minutes * 60)
        recent = [s for s in self._history if s.timestamp >= cutoff]

        if not recent:
            return {"samples": 0}

        return {
            "samples": len(recent),
            "period_minutes": minutes,
            "cpu_avg": round(sum(s.cpu_percent for s in recent) / len(recent), 1),
            "memory_avg_pct": round(
                sum(s.memory_percent for s in recent) / len(recent), 1
            ),
            "disk_latest_pct": round(recent[-1].disk_percent, 1),
        }
