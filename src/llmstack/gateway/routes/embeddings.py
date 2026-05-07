"""POST /v1/embeddings — OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llmstack.gateway.proxy import proxy_embeddings

router = APIRouter()


@router.post("/embeddings")
async def embeddings(request: Request):
    payload = await request.json()
    result = await proxy_embeddings(payload)
    return JSONResponse(content=result)
