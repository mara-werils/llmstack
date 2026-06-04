"""Health monitor — comprehensive system health checks with alerting."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    latency_ms: float
    message: str = ""
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemHealth:
    """Overall system health."""

    status: HealthStatus
    checks: list[HealthCheck]
    uptime_seconds: float
    timestamp: float


class HealthMonitor:
    """Monitor system health with periodic checks."""

    def __init__(self):
        self.start_time = time.time()
        self._last_checks: list[HealthCheck] = []
        self._alert_callbacks: list = []
        self._check_history: list[SystemHealth] = []
        self._max_history = 100

    def register_alert(self, callback) -> None:
        """Register a callback for health alerts."""
        self._alert_callbacks.append(callback)

    async def check_all(
        self,
        ollama_url: str = "http://localhost:11434",
        redis_url: str | None = None,
        qdrant_url: str | None = None,
    ) -> SystemHealth:
        """Run all health checks."""
        checks = await asyncio.gather(
            self._check_ollama(ollama_url),
            self._check_disk(),
            self._check_memory(),
            self._check_redis(redis_url) if redis_url else self._skip_check("redis"),
            self._check_qdrant(qdrant_url) if qdrant_url else self._skip_check("qdrant"),
            return_exceptions=True,
        )

        valid_checks = []
        for c in checks:
            if isinstance(c, HealthCheck):
                valid_checks.append(c)
            elif isinstance(c, Exception):
                valid_checks.append(
                    HealthCheck(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=0,
                        message=str(c),
                    )
                )

        # Determine overall status
        statuses = [c.status for c in valid_checks]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        health = SystemHealth(
            status=overall,
            checks=valid_checks,
            uptime_seconds=time.time() - self.start_time,
            timestamp=time.time(),
        )

        self._last_checks = valid_checks
        self._check_history.append(health)
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history :]

        # Fire alerts for unhealthy checks
        for check in valid_checks:
            if check.status == HealthStatus.UNHEALTHY:
                for cb in self._alert_callbacks:
                    try:
                        cb(check)
                    except Exception:
                        pass

        return health

    async def _check_ollama(self, url: str) -> HealthCheck:
        """Check Ollama health."""
        import httpx

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{url}/api/version")
                latency = (time.time() - start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    return HealthCheck(
                        name="ollama",
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency,
                        message=f"v{data.get('version', 'unknown')}",
                        details=data,
                    )
                return HealthCheck(
                    name="ollama",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message=f"HTTP {resp.status_code}",
                )
        except Exception as e:
            return HealthCheck(
                name="ollama",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_disk(self) -> HealthCheck:
        """Check disk space."""
        import shutil

        start = time.time()
        try:
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            pct_used = (usage.used / usage.total) * 100
            latency = (time.time() - start) * 1000

            if pct_used > 95:
                status = HealthStatus.UNHEALTHY
            elif pct_used > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheck(
                name="disk",
                status=status,
                latency_ms=latency,
                message=f"{free_gb:.1f}GB free ({pct_used:.0f}% used)",
                details={"free_gb": free_gb, "total_gb": total_gb, "pct_used": pct_used},
            )
        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message=str(e),
            )

    async def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        start = time.time()
        try:
            import subprocess
            import platform

            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    stats = {}
                    for line in lines:
                        if ":" in line:
                            key, _, val = line.partition(":")
                            val = val.strip().rstrip(".")
                            try:
                                stats[key.strip()] = int(val)
                            except ValueError:
                                pass

                    page_size = 16384  # Apple Silicon default
                    free_pages = stats.get("Pages free", 0) + stats.get("Pages speculative", 0)
                    free_gb = (free_pages * page_size) / (1024**3)

                    status = (
                        HealthStatus.HEALTHY
                        if free_gb > 2
                        else (HealthStatus.DEGRADED if free_gb > 0.5 else HealthStatus.UNHEALTHY)
                    )
                    return HealthCheck(
                        name="memory",
                        status=status,
                        latency_ms=(time.time() - start) * 1000,
                        message=f"{free_gb:.1f}GB available",
                        details={"free_gb": free_gb},
                    )

            elif platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])

                total = meminfo.get("MemTotal", 0) / (1024**2)
                available = meminfo.get("MemAvailable", 0) / (1024**2)
                pct_used = ((total - available) / max(1, total)) * 100

                status = (
                    HealthStatus.HEALTHY
                    if pct_used < 85
                    else (HealthStatus.DEGRADED if pct_used < 95 else HealthStatus.UNHEALTHY)
                )
                return HealthCheck(
                    name="memory",
                    status=status,
                    latency_ms=(time.time() - start) * 1000,
                    message=f"{available:.1f}GB available ({pct_used:.0f}% used)",
                    details={"total_gb": total, "available_gb": available},
                )

            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="Unsupported platform",
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message=str(e),
            )

    async def _check_redis(self, url: str) -> HealthCheck:
        """Check Redis connectivity."""
        start = time.time()
        try:
            import redis

            r = redis.from_url(url, socket_timeout=3)
            r.ping()
            info = r.info("server")
            return HealthCheck(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f"v{info.get('redis_version', 'unknown')}",
            )
        except Exception as e:
            return HealthCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_qdrant(self, url: str) -> HealthCheck:
        """Check Qdrant connectivity."""
        import httpx

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{url}/healthz")
                latency = (time.time() - start) * 1000
                if resp.status_code == 200:
                    return HealthCheck(
                        name="qdrant",
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency,
                        message="OK",
                    )
                return HealthCheck(
                    name="qdrant",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message=f"HTTP {resp.status_code}",
                )
        except Exception as e:
            return HealthCheck(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _skip_check(self, name: str) -> HealthCheck:
        """Skip a check (service not configured)."""
        return HealthCheck(
            name=name,
            status=HealthStatus.UNKNOWN,
            latency_ms=0,
            message="Not configured",
        )

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent health check history."""
        return [
            {
                "status": h.status.value,
                "timestamp": h.timestamp,
                "uptime": h.uptime_seconds,
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "latency_ms": c.latency_ms,
                        "message": c.message,
                    }
                    for c in h.checks
                ],
            }
            for h in self._check_history[-limit:]
        ]
