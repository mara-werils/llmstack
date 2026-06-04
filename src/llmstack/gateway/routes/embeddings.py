"""POST /v1/embeddings — OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from llmstack.gateway.proxy import proxy_embeddings
from llmstack.gateway.schemas import EmbeddingRequest

router = APIRouter()


@router.post("/embeddings")
async def embeddings(request: Request):
    try:
        raw = await request.json()
        validated = EmbeddingRequest.model_validate(raw)
        payload = validated.model_dump(exclude_none=True)
    except ValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": str(exc),
                    "type": "validation_error",
                }
            },
        )
    result = await proxy_embeddings(payload)
    return JSONResponse(content=result)
