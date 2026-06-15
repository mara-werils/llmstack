"""Tests for the gateway HealthMonitor."""

from __future__ import annotations

import httpx
import pytest

from llmstack.gateway.health import (
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    SystemHealth,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClient:
    """Stand-in for httpx.AsyncClient driven by a response map or exception."""

    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if self._exc:
            raise self._exc
        return self._response


def _patch_httpx(monkeypatch, **kw):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(**kw))


@pytest.fixture
def monitor():
    return HealthMonitor()


class TestProperties:
    def test_is_healthy_when_no_history(self, monitor):
        assert monitor.is_healthy is True
        assert monitor.check_count == 0

    def test_uptime_positive(self, monitor):
        assert monitor.uptime_seconds >= 0

    def test_is_healthy_reflects_last_check(self, monitor):
        monitor._check_history.append(
            SystemHealth(status=HealthStatus.UNHEALTHY, checks=[], uptime_seconds=1, timestamp=1)
        )
        assert monitor.is_healthy is False


class TestCheckAll:
    async def _stub_checks(self, monitor, status):
        async def ok(*a, **k):
            return HealthCheck(name="ollama", status=status, latency_ms=1.0)

        for name in ("_check_ollama", "_check_disk", "_check_memory"):
            setattr(monitor, name, ok)

    async def test_overall_healthy(self, monitor):
        await self._stub_checks(monitor, HealthStatus.HEALTHY)
        result = await monitor.check_all()
        assert result.status == HealthStatus.HEALTHY
        assert monitor.check_count == 1

    async def test_overall_degraded(self, monitor):
        await self._stub_checks(monitor, HealthStatus.DEGRADED)
        result = await monitor.check_all()
        assert result.status == HealthStatus.DEGRADED

    async def test_overall_unhealthy_fires_alert(self, monitor):
        await self._stub_checks(monitor, HealthStatus.UNHEALTHY)
        fired = []
        monitor.register_alert(fired.append)
        monitor.register_alert(lambda c: (_ for _ in ()).throw(RuntimeError("bad cb")))
        result = await monitor.check_all()
        assert result.status == HealthStatus.UNHEALTHY
        assert len(fired) == 3  # ollama, disk, memory all unhealthy

    async def test_exception_in_check_becomes_unhealthy(self, monitor):
        async def boom(*a, **k):
            raise ValueError("kaboom")

        async def ok(*a, **k):
            return HealthCheck(name="disk", status=HealthStatus.HEALTHY, latency_ms=1.0)

        monitor._check_ollama = boom
        monitor._check_disk = ok
        monitor._check_memory = ok
        result = await monitor.check_all()
        assert result.status == HealthStatus.UNHEALTHY
        assert any(c.message == "kaboom" for c in result.checks)

    async def test_history_capped(self, monitor):
        await self._stub_checks(monitor, HealthStatus.HEALTHY)
        monitor._max_history = 3
        for _ in range(5):
            await monitor.check_all()
        assert monitor.check_count == 3

    async def test_optional_services_skipped_by_default(self, monitor):
        await self._stub_checks(monitor, HealthStatus.HEALTHY)
        result = await monitor.check_all()
        skipped = [c for c in result.checks if c.message == "Not configured"]
        assert {c.name for c in skipped} == {"redis", "qdrant"}


class TestIndividualChecks:
    async def test_ollama_healthy(self, monitor, monkeypatch):
        _patch_httpx(monkeypatch, response=_FakeResponse(200, {"version": "0.1.2"}))
        c = await monitor._check_ollama("http://x")
        assert c.status == HealthStatus.HEALTHY
        assert "0.1.2" in c.message

    async def test_ollama_degraded(self, monitor, monkeypatch):
        _patch_httpx(monkeypatch, response=_FakeResponse(503))
        c = await monitor._check_ollama("http://x")
        assert c.status == HealthStatus.DEGRADED

    async def test_ollama_unhealthy(self, monitor, monkeypatch):
        _patch_httpx(monkeypatch, exc=httpx.ConnectError("refused"))
        c = await monitor._check_ollama("http://x")
        assert c.status == HealthStatus.UNHEALTHY

    async def test_qdrant_healthy(self, monitor, monkeypatch):
        _patch_httpx(monkeypatch, response=_FakeResponse(200))
        c = await monitor._check_qdrant("http://x")
        assert c.status == HealthStatus.HEALTHY

    async def test_qdrant_degraded(self, monitor, monkeypatch):
        _patch_httpx(monkeypatch, response=_FakeResponse(500))
        c = await monitor._check_qdrant("http://x")
        assert c.status == HealthStatus.DEGRADED

    async def test_qdrant_unhealthy(self, monitor, monkeypatch):
        _patch_httpx(monkeypatch, exc=httpx.ConnectError("nope"))
        c = await monitor._check_qdrant("http://x")
        assert c.status == HealthStatus.UNHEALTHY

    async def test_redis_unhealthy_on_bad_url(self, monitor):
        c = await monitor._check_redis("redis://127.0.0.1:1")
        assert c.status == HealthStatus.UNHEALTHY

    async def test_disk_check_runs(self, monitor):
        c = await monitor._check_disk()
        assert c.name == "disk"
        assert c.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY}
        assert "free_gb" in c.details

    async def test_memory_check_runs(self, monitor):
        c = await monitor._check_memory()
        assert c.name == "memory"
        assert isinstance(c.status, HealthStatus)

    async def test_memory_unsupported_platform(self, monitor, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Plan9")
        c = await monitor._check_memory()
        assert c.status == HealthStatus.UNKNOWN
        assert "Unsupported" in c.message

    async def test_skip_check(self, monitor):
        c = await monitor._skip_check("redis")
        assert c.status == HealthStatus.UNKNOWN
        assert c.message == "Not configured"


class TestHistory:
    async def test_get_history_shape(self, monitor):
        async def ok(*a, **k):
            return HealthCheck(name="x", status=HealthStatus.HEALTHY, latency_ms=2.0, message="m")

        monitor._check_ollama = ok
        monitor._check_disk = ok
        monitor._check_memory = ok
        await monitor.check_all()
        hist = monitor.get_history(limit=10)
        assert len(hist) == 1
        assert hist[0]["status"] == "healthy"
        assert hist[0]["checks"][0]["name"] in {"x", "redis", "qdrant"}
