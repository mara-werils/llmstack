"""GET /healthz — gateway health check with full system status."""

from __future__ import annotations

import os
import time

import httpx
from fastapi import APIRouter

from llmstack import __version__
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter()

INFERENCE_URL = os.getenv("LLMSTACK_INFERENCE_URL", "")
QDRANT_URL = os.getenv("LLMSTACK_QDRANT_URL", "")
REDIS_URL = os.getenv("LLMSTACK_REDIS_URL", "")

_START_TIME = time.monotonic()


def _inference_base(url: str) -> str:
    """Server base for inference health probes, stripping a trailing ``/v1``.

    Use ``removesuffix`` rather than ``replace('/v1', '')`` so a ``/v1`` that
    appears earlier in the URL (a host like ``v1.example.com`` or a longer path)
    is left intact and only the OpenAI-style ``/v1`` suffix is removed.
    """
    return url.rstrip("/").removesuffix("/v1")


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
        health_url = _inference_base(INFERENCE_URL)
        checks["inference"] = await _check_url(health_url) or await _check_url(
            health_url + "/health"
        )

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

    try:
        from llmstack.gateway.router._state import get_stats

        stats = get_stats()
        if stats is not None:
            summary = stats.summary()
            extras["router"] = {
                "total_requests": summary["total_requests"],
                "tier_distribution": summary["tier_distribution"],
                "provider_distribution": summary.get("provider_distribution", {}),
                "estimated_savings_pct": summary["estimated_savings_pct"],
                "total_cost_usd": summary.get("total_cost_usd", 0.0),
                "cost_by_provider_usd": summary.get("cost_by_provider_usd", {}),
            }
    except Exception:
        pass

    try:
        from llmstack.gateway.providers.registry import get_registry

        registry = get_registry()
        if registry is not None:
            providers_status = {}
            for name, p in registry.all_providers().items():
                providers_status[name] = {
                    "models": len([m for m in registry.all_models() if m.provider == name]),
                }
            extras["providers"] = providers_status
    except Exception:
        pass

    uptime_s = time.monotonic() - _START_TIME

    return JSONResponse(
        content={
            "status": "ok" if all_ok else "degraded",
            "version": __version__,
            "uptime_s": round(uptime_s, 1),
            "services": checks,
            **extras,
        },
        status_code=status_code,
    )


@router.get("/ping")
async def ping():
    """Ultra-lightweight ping — no backend checks, just returns pong."""
    return PlainTextResponse("pong", status_code=200)


@router.get("/healthz/live")
async def liveness():
    """Kubernetes liveness probe — always returns 200 if the process is alive."""
    return JSONResponse(content={"status": "alive"}, status_code=200)


@router.get("/health")
async def health_alias():
    """Convenience alias of the liveness probe (`/healthz/live`).

    Many tools (Docker HEALTHCHECK, uptime monitors, editor extensions) probe
    `/health` by convention; mirroring liveness here means they just work.
    """
    return JSONResponse(content={"status": "alive"}, status_code=200)


@router.get("/healthz/ready")
async def readiness():
    """Kubernetes readiness probe — returns 200 only if all backends are reachable."""
    checks = {}

    if INFERENCE_URL:
        health_url = _inference_base(INFERENCE_URL)
        checks["inference"] = await _check_url(health_url) or await _check_url(
            health_url + "/health"
        )

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
    return JSONResponse(
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
        status_code=200 if all_ok else 503,
    )


@router.get("/metrics")
async def metrics():
    from llmstack.gateway.middleware.metrics import get_prometheus_metrics

    return PlainTextResponse(
        content=get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
