"""GET /healthz — gateway health check with full system status."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter()

INFERENCE_URL = os.getenv("LLMSTACK_INFERENCE_URL", "")
QDRANT_URL = os.getenv("LLMSTACK_QDRANT_URL", "")
REDIS_URL = os.getenv("LLMSTACK_REDIS_URL", "")


async def _check_url(url: str) -> bool:
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


@router.get("/healthz")
async def healthz():
    checks = {}

    if INFERENCE_URL:
        # Ollama uses / as health, vLLM uses /health
        health_url = INFERENCE_URL.replace("/v1", "")
        checks["inference"] = await _check_url(health_url) or await _check_url(health_url + "/health")

    if QDRANT_URL:
        checks["qdrant"] = await _check_url(f"{QDRANT_URL}/healthz")

    if REDIS_URL:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(REDIS_URL, socket_connect_timeout=3)
            await r.ping()
            checks["redis"] = True
            await r.aclose()
        except Exception:
            checks["redis"] = False

    all_ok = all(checks.values()) if checks else True
    status_code = 200 if all_ok else 503

    # Include circuit breaker and cache stats
    extras = {}
    try:
        from llmstack.gateway.circuit_breaker import get_inference_breaker
        extras["circuit_breaker"] = get_inference_breaker().metrics()
    except Exception:
        pass

    try:
        from llmstack.gateway.cache import _cache
        if _cache is not None:
            extras["cache"] = _cache.stats.to_dict()
    except Exception:
        pass

    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "services": checks, **extras},
        status_code=status_code,
    )


@router.get("/metrics")
async def metrics():
    from llmstack.gateway.middleware.metrics import get_prometheus_metrics
    return PlainTextResponse(
        content=get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
