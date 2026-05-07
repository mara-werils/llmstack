"""POST /v1/chat/completions — OpenAI-compatible chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llmstack.gateway.circuit_breaker import CircuitBreakerError
from llmstack.gateway.proxy import proxy_chat_completion

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    stream = payload.get("stream", False)

    try:
        if stream:
            chunks = await proxy_chat_completion(payload, stream=True)
            return StreamingResponse(
                chunks,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            result = await proxy_chat_completion(payload, stream=False)
            # Indicate cache hit in response headers
            response = JSONResponse(content=result)
            if isinstance(result, dict) and result.get("_cached"):
                response.headers["X-Cache"] = "HIT"
                response.headers["X-Cache-Age"] = str(result.pop("_cache_age_s", 0))
                result.pop("_cached", None)
                result.pop("_cached_at", None)
            else:
                response.headers["X-Cache"] = "MISS"
            return response

    except CircuitBreakerError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "Inference backend is temporarily unavailable",
                    "type": "service_unavailable",
                    "retry_after": round(exc.retry_after),
                }
            },
            headers={"Retry-After": str(round(exc.retry_after))},
        )
