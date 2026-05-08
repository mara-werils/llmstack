"""Router debug / analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/router/stats")
async def router_stats(request: Request):
    """Return routing statistics (model distribution, savings, latencies)."""
    from llmstack.gateway.router import _stats  # noqa: WPS433

    stats = _get_stats()
    if stats is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Router is not enabled", "type": "not_found"}},
        )
    return JSONResponse(content=stats.summary())


@router.post("/router/classify")
async def router_classify(request: Request):
    """Classify a query without routing — useful for debugging.

    Body: ``{"messages": [...]}``
    Returns the QueryProfile.
    """
    from llmstack.gateway.router import _router  # noqa: WPS433

    rtr = _get_router()
    if rtr is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Router is not enabled", "type": "not_found"}},
        )

    body = await request.json()
    messages = body.get("messages", [])
    profile = rtr.classify_only(messages)

    return JSONResponse(content={
        "score": profile.score,
        "tier": profile.tier,
        "factors": profile.factors,
        "suggested_model": profile.suggested_model,
    })


# ---------------------------------------------------------------------------
# Helpers — import the module-level singletons lazily
# ---------------------------------------------------------------------------

def _get_stats():
    try:
        from llmstack.gateway.router._state import get_stats
        return get_stats()
    except Exception:
        return None


def _get_router():
    try:
        from llmstack.gateway.router._state import get_router
        return get_router()
    except Exception:
        return None
