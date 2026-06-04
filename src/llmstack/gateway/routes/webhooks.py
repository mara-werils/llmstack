"""Webhook management API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llmstack.gateway.webhooks import WebhookManager, WebhookEvent

router = APIRouter(tags=["Webhooks"])

_manager: WebhookManager | None = None


def get_manager() -> WebhookManager:
    global _manager
    if _manager is None:
        _manager = WebhookManager()
    return _manager


class RegisterWebhookRequest(BaseModel):
    url: str
    events: list[str]
    secret: str = ""
    description: str = ""
    headers: dict[str, str] = Field(default_factory=dict)


@router.get("/webhooks")
async def list_webhooks():
    """List all registered webhook endpoints."""
    mgr = get_manager()
    endpoints = mgr.list_endpoints()
    return {"endpoints": [e.to_dict() for e in endpoints]}


@router.post("/webhooks", status_code=201)
async def register_webhook(req: RegisterWebhookRequest):
    """Register a new webhook endpoint."""
    mgr = get_manager()
    events = []
    for e in req.events:
        try:
            events.append(WebhookEvent(e))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event: {e}. Valid: {[v.value for v in WebhookEvent]}",
            )
    endpoint = mgr.register(
        url=req.url,
        events=events,
        secret=req.secret,
        description=req.description,
        headers=req.headers,
    )
    return endpoint.to_dict()


@router.delete("/webhooks/{endpoint_id}")
async def unregister_webhook(endpoint_id: str):
    """Remove a webhook endpoint."""
    mgr = get_manager()
    if not mgr.unregister(endpoint_id):
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"deleted": True}


@router.get("/webhooks/{endpoint_id}/deliveries")
async def list_deliveries(endpoint_id: str, limit: int = 50):
    """List recent delivery attempts for a webhook."""
    mgr = get_manager()
    deliveries = mgr.get_deliveries(endpoint_id=endpoint_id, limit=limit)
    return {"deliveries": [d.to_dict() for d in deliveries]}


@router.get("/webhooks/stats")
async def webhook_stats():
    """Get webhook system statistics."""
    return get_manager().get_stats()
