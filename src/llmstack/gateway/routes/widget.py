"""Widget embedding routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(tags=["Widget"])


@router.get("/widget/embed", response_class=HTMLResponse, include_in_schema=False)
async def widget_embed() -> HTMLResponse:
    return HTMLResponse(
        "<!DOCTYPE html><html><head><title>LLMStack Chat</title></head>"
        "<body><h2>LLMStack Chat Widget Demo</h2>"
        "<p>The chat widget appears in the bottom-right corner.</p>"
        '<script src="/ui/widget.js"></script>'
        "</body></html>"
    )


@router.get("/widget/config")
async def widget_config() -> JSONResponse:
    return JSONResponse(
        {
            "name": "LLMStack",
            "version": "1.0.0",
            "chat_endpoint": "/v1/chat/completions",
            "models_endpoint": "/v1/models",
        }
    )
