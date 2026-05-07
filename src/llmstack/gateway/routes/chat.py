"""POST /v1/chat/completions — OpenAI-compatible chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llmstack.gateway.proxy import proxy_chat_completion

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    stream = payload.get("stream", False)

    if stream:
        chunks = await proxy_chat_completion(payload, stream=True)
        return StreamingResponse(
            chunks,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        result = await proxy_chat_completion(payload, stream=False)
        return JSONResponse(content=result)
