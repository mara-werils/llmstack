"""GET /v1/models — list available models."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from llmstack.gateway.proxy import proxy_models

router = APIRouter()


@router.get("/models")
async def list_models():
    result = await proxy_models()
    return JSONResponse(content=result)
