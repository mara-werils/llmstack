"""GET /healthz — gateway health check."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

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
        checks["inference"] = await _check_url(INFERENCE_URL.replace("/v1", "/health"))

    if QDRANT_URL:
        checks["qdrant"] = await _check_url(f"{QDRANT_URL}/healthz")

    all_ok = all(checks.values()) if checks else True
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "services": checks},
        status_code=status_code,
    )


@router.get("/metrics")
async def metrics():
    from llmstack.gateway.middleware.metrics import get_prometheus_metrics
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
