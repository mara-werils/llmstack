"""Tests for the model warm-up runner and report aggregation."""

from __future__ import annotations

import asyncio

from llmstack.gateway.warmup import (
    WarmupConfig,
    WarmupManager,
    WarmupReport,
    WarmupResult,
    warmup_all,
    warmup_model,
)


class TestWarmupReport:
    def test_empty_report_properties(self):
        r = WarmupReport()
        assert r.success_count == 0
        assert r.failure_count == 0
        assert r.avg_latency_ms == 0.0
        assert r.max_latency_ms == 0.0
        assert r.success_rate == 0.0
        assert r.started_at > 0

    def test_aggregates(self):
        r = WarmupReport(
            results=[
                WarmupResult(model="a", success=True, latency_ms=10.0),
                WarmupResult(model="b", success=False, latency_ms=30.0),
            ]
        )
        assert r.success_count == 1
        assert r.failure_count == 1
        assert r.avg_latency_ms == 20.0
        assert r.max_latency_ms == 30.0
        assert r.success_rate == 0.5
        d = r.to_dict()
        assert d["total_models"] == 2
        assert d["succeeded"] == 1
        assert len(d["results"]) == 2


class TestWarmupModel:
    async def test_success(self):
        async def handler(payload):
            assert payload["temperature"] == 0
            return {"ok": True}

        res = await warmup_model("llama3", handler, provider="local")
        assert res.success is True
        assert res.provider == "local"
        assert res.error == ""

    async def test_timeout(self):
        async def slow(payload):
            await asyncio.sleep(1)

        res = await warmup_model("m", slow, timeout=0.01)
        assert res.success is False
        assert "Timeout" in res.error

    async def test_handler_exception(self):
        async def boom(payload):
            raise ValueError("nope")

        res = await warmup_model("m", boom)
        assert res.success is False
        assert res.error == "nope"

    async def test_uses_config_overrides(self):
        seen = {}

        async def handler(payload):
            seen.update(payload)

        cfg = WarmupConfig(prompt=[{"role": "user", "content": "ping"}], max_tokens=5)
        await warmup_model("m", handler, config=cfg)
        assert seen["max_tokens"] == 5
        assert seen["messages"][0]["content"] == "ping"


class TestWarmupAll:
    async def test_warms_all_models(self):
        async def handler(payload):
            return {"ok": True}

        models = [{"name": "a", "provider": "local"}, {"name": "b"}]
        report = await warmup_all(models, handler, concurrency=2)
        assert report.success_count == 2
        assert report.total_time_ms >= 0

    async def test_mixed_success_and_failure(self):
        async def handler(payload):
            if payload["model"] == "bad":
                raise RuntimeError("fail")
            return {"ok": True}

        models = [{"name": "good"}, {"name": "bad"}]
        report = await warmup_all(models, handler)
        assert report.success_count == 1
        assert report.failure_count == 1


class TestWarmupManager:
    def test_init_copies_models(self):
        models = ["a", "b"]
        mgr = WarmupManager(models)
        models.append("c")
        assert mgr.models == ["a", "b"]
        assert isinstance(mgr.config, WarmupConfig)
