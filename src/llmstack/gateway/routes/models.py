"""GET /v1/models — list available models from all providers."""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from llmstack.gateway.proxy import proxy_models

router = APIRouter()


@router.get("/models")
async def list_models():
    # Start with local models
    try:
        result = await proxy_models()
    except Exception:
        result = {"object": "list", "data": []}

    local_models = result.get("data", [])

    # Add models from all registered providers
    try:
        from llmstack.gateway.providers.registry import get_registry

        registry = get_registry()
        if registry is not None:
            for pm in registry.all_models():
                local_models.append(
                    {
                        "id": pm.id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": pm.provider,
                        "context_length": pm.context_length,
                        "x_llmstack": {
                            "provider": pm.provider,
                            "cost_per_m_input": pm.cost_per_m_input,
                            "cost_per_m_output": pm.cost_per_m_output,
                        },
                    }
                )
    except Exception:
        pass

    # Deduplicate by model ID
    seen = set()
    unique = []
    for m in local_models:
        mid = m.get("id", "")
        if mid not in seen:
            seen.add(mid)
            unique.append(m)

    return JSONResponse(content={"object": "list", "data": unique})
