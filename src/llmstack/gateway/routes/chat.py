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


def _try_route(payload: dict) -> tuple[dict, str | None, str | None, str | None]:
    """Attempt to route the request via the smart model router.

    Returns ``(payload, routed_model, tier, provider)`` — the payload is
    mutated in-place only when the router selects a different model.
    """
    try:
        from llmstack.gateway.router._state import get_router
    except Exception:
        return payload, None, None, None

    rtr = get_router()
    if rtr is None:
        return payload, None, None, None

    messages = payload.get("messages", [])
    request_model = payload.get("model")

    # "auto" or empty model means "let the router decide"
    known_models = {m.name for m in rtr.models}
    is_auto = not request_model or request_model == "auto"

    # If the user explicitly set a model that is NOT in our tier list,
    # try to find it in the provider registry
    if not is_auto and request_model not in known_models:
        provider = _resolve_provider_for_model(request_model)
        return payload, None, None, provider

    # If user explicitly picked a known model, set override for this request
    if not is_auto and request_model in known_models:
        rtr.override(request_model)

    decision = rtr.route(messages)

    # Clear one-shot override
    rtr.override(None)

    # Replace model in payload
    payload["model"] = decision.model

    logger.info(
        "Router: tier=%s score=%.3f model=%s provider=%s speedup=%.1fx",
        decision.profile.tier,
        decision.profile.score,
        decision.model,
        decision.provider,
        decision.estimated_speedup,
    )

    return payload, decision.model, decision.profile.tier, decision.provider


def _resolve_provider_for_model(model_id: str) -> str | None:
    """Check if a model belongs to a registered provider."""
    try:
        from llmstack.gateway.providers.registry import get_registry
        registry = get_registry()
        if registry is None:
            return None
        provider = registry.get_provider_for_model(model_id)
        if provider:
            return provider.name
        # Try guessing by prefix
        provider = registry._guess_provider(model_id)
        return provider.name if provider else None
    except Exception:
        return None


def _record_trace(
    payload: dict, result: dict, model: str, provider: str | None,
    tier: str | None, latency_ms: float, cost_usd: float,
    cached: bool = False,
) -> None:
    """Record a trace with quality scoring for observability."""
    try:
        from llmstack.observe._state import get_trace_store, get_scorer, get_tracker
        from llmstack.observe.traces import Trace

        store = get_trace_store()
        scorer = get_scorer()
        if store is None:
            return

        # Extract response content
        response_text = ""
        finish_reason = ""
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
                finish_reason = choices[0].get("finish_reason", "")

        # Extract usage
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Extract query for scoring
        messages = payload.get("messages", [])
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break

        # Score quality
        quality = {}
        if scorer and response_text:
            score = scorer.score(query, response_text)
            quality = score.to_dict()

        trace = Trace(
            model=model,
            provider=provider or "local",
            messages=messages,
            temperature=payload.get("temperature", 0.0),
            stream=payload.get("stream", False),
            routed_model=model,
            routed_tier=tier or "",
            response=response_text[:500],
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cached=cached,
            quality=quality,
        )
        store.add(trace)

        # Track quality for drift detection
        tracker = get_tracker()
        if tracker and quality:
            tracker.record(quality, model=model, provider=provider or "local")

    except Exception:
        pass


def _record_stats(
    model: str | None, tier: str | None, latency_ms: float,
    provider: str | None = None, cost_usd: float = 0.0,
) -> None:
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
        decision = RoutingDecision(
            model=model, profile=profile, provider=provider or "local",
        )
        stats.record(decision, latency_ms, cost_usd=cost_usd)
    except Exception:
        pass


@router.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    stream = payload.get("stream", False)

    # Smart routing (now returns provider too)
    payload, routed_model, tier, provider = _try_route(payload)

    t0 = time.monotonic()

    try:
        if stream:
            chunks = await proxy_chat_completion(payload, stream=True, provider_name=provider)
            headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            if routed_model:
                headers["X-Model-Router"] = routed_model
            if tier:
                headers["X-Query-Tier"] = tier
            if provider:
                headers["X-Provider"] = provider
            return StreamingResponse(
                chunks,
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            result = await proxy_chat_completion(
                payload, stream=False, provider_name=provider,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Extract cost from provider response
            cost_usd = 0.0
            if isinstance(result, dict):
                x_info = result.get("x_llmstack", {})
                cost_usd = x_info.get("cost_usd", 0.0)

            _record_stats(routed_model, tier, elapsed_ms, provider, cost_usd)

            # Trace and quality scoring
            _record_trace(
                payload, result, routed_model or payload.get("model", ""),
                provider, tier, elapsed_ms, cost_usd,
                cached=isinstance(result, dict) and result.get("_cached", False),
            )

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
            if provider:
                response.headers["X-Provider"] = provider
            if cost_usd > 0:
                response.headers["X-Cost-USD"] = f"{cost_usd:.6f}"

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
