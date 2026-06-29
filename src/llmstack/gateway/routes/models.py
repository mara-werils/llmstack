"""GET /v1/models — list available models from all providers."""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from llmstack.gateway.proxy import proxy_models

router = APIRouter()

# Stable "created" timestamp for provider-registry models. OpenAI's schema treats
# `created` as the model's creation time, not "now", so it must not change between
# calls -- clients diff model lists and a moving timestamp makes every entry look
# changed on every poll. Stamp it once at process start.
_REGISTRY_MODEL_CREATED = int(time.time())


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
                        "created": _REGISTRY_MODEL_CREATED,
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
