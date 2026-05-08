"""POST /v1/chat/completions — OpenAI-compatible chat endpoint."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llmstack.gateway.circuit_breaker import CircuitBreakerError
from llmstack.gateway.proxy import proxy_chat_completion

logger = logging.getLogger(__name__)

router = APIRouter()


def _try_route(payload: dict) -> tuple[dict, str | None, str | None]:
    """Attempt to route the request via the smart model router.

    Returns ``(payload, routed_model, tier)`` — the payload is mutated
    in-place only when the router selects a different model.
    """
    try:
        from llmstack.gateway.router._state import get_router
    except Exception:
        return payload, None, None

    rtr = get_router()
    if rtr is None:
        return payload, None, None

    messages = payload.get("messages", [])
    request_model = payload.get("model")

    # If the user explicitly set a model that is NOT in our tier list,
    # pass through without routing.
    known_models = {m.name for m in rtr.models}
    if request_model and request_model not in known_models:
        return payload, None, None

    # If user explicitly picked a known model, set override for this request
    if request_model and request_model in known_models:
        rtr.override(request_model)

    decision = rtr.route(messages)

    # Clear one-shot override
    rtr.override(None)

    # Replace model in payload
    payload["model"] = decision.model

    logger.info(
        "Router: tier=%s score=%.3f model=%s speedup=%.1fx",
        decision.profile.tier,
        decision.profile.score,
        decision.model,
        decision.estimated_speedup,
    )

    return payload, decision.model, decision.profile.tier


def _record_stats(model: str | None, tier: str | None, latency_ms: float) -> None:
    """Record routing stats if the router is active."""
    if model is None:
        return
    try:
        from llmstack.gateway.router._state import get_stats
        from llmstack.gateway.router.router import RoutingDecision
        from llmstack.gateway.router.classifier import QueryProfile

        stats = get_stats()
        if stats is None:
            return

        # Build a minimal decision for recording
        profile = QueryProfile(score=0.0, tier=tier or "simple", factors={})
        decision = RoutingDecision(model=model, profile=profile)
        stats.record(decision, latency_ms)
    except Exception:
        pass


@router.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    stream = payload.get("stream", False)

    # Smart routing
    payload, routed_model, tier = _try_route(payload)

    t0 = time.monotonic()

    try:
        if stream:
            chunks = await proxy_chat_completion(payload, stream=True)
            headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            if routed_model:
                headers["X-Model-Router"] = routed_model
            if tier:
                headers["X-Query-Tier"] = tier
            return StreamingResponse(
                chunks,
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            result = await proxy_chat_completion(payload, stream=False)
            elapsed_ms = (time.monotonic() - t0) * 1000
            _record_stats(routed_model, tier, elapsed_ms)

            # Indicate cache hit in response headers
            response = JSONResponse(content=result)
            if isinstance(result, dict) and result.get("_cached"):
                response.headers["X-Cache"] = "HIT"
                response.headers["X-Cache-Age"] = str(result.pop("_cache_age_s", 0))
                result.pop("_cached", None)
                result.pop("_cached_at", None)
            else:
                response.headers["X-Cache"] = "MISS"

            if routed_model:
                response.headers["X-Model-Router"] = routed_model
            if tier:
                response.headers["X-Query-Tier"] = tier

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
