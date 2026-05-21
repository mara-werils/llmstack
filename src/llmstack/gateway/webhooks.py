"""Webhook notification system — event-driven notifications for LLM operations.

Supports configurable webhook endpoints that receive POST notifications
for events like: budget alerts, quality drops, errors, model switches, etc.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Events that can trigger webhook notifications."""

    REQUEST_COMPLETED = "request.completed"
    REQUEST_ERROR = "request.error"
    BUDGET_ALERT = "budget.alert"
    BUDGET_EXCEEDED = "budget.exceeded"
    QUALITY_DROP = "quality.drop"
    MODEL_FALLBACK = "model.fallback"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker.open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker.close"
    CACHE_EVICTION = "cache.eviction"
    RATE_LIMIT_HIT = "rate_limit.hit"


@dataclass
class WebhookEndpoint:
    """A registered webhook endpoint."""

    id: str = ""
    url: str = ""
    events: list[WebhookEvent] = field(default_factory=list)
    secret: str = ""
    active: bool = True
    description: str = ""
    created_at: float = 0.0
    failure_count: int = 0
    last_triggered: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "active": self.active,
            "description": self.description,
            "created_at": self.created_at,
            "failure_count": self.failure_count,
            "last_triggered": self.last_triggered,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    id: str = ""
    endpoint_id: str = ""
    event: str = ""
    payload: dict = field(default_factory=dict)
    status_code: int = 0
    success: bool = False
    error: str = ""
    timestamp: float = 0.0
    duration_ms: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "endpoint_id": self.endpoint_id,
            "event": self.event,
            "status_code": self.status_code,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 1),
        }


def _compute_signature(payload: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for webhook payload verification."""
    return hmac.new(
        secret.encode(), payload.encode(), hashlib.sha256,
    ).hexdigest()


class WebhookManager:
    """Manages webhook endpoints and dispatches event notifications."""

    MAX_FAILURES = 10  # disable endpoint after this many consecutive failures
    MAX_DELIVERIES = 1000  # keep last N delivery records

    def __init__(self):
        self._lock = Lock()
        self._endpoints: dict[str, WebhookEndpoint] = {}
        self._deliveries: list[WebhookDelivery] = []
        self._event_counts: dict[str, int] = {}

    def register(
        self,
        url: str,
        events: list[WebhookEvent],
        secret: str = "",
        description: str = "",
        headers: dict[str, str] | None = None,
    ) -> WebhookEndpoint:
        """Register a new webhook endpoint."""
        endpoint = WebhookEndpoint(
            url=url,
            events=events,
            secret=secret,
            description=description,
            headers=headers or {},
        )
        with self._lock:
            self._endpoints[endpoint.id] = endpoint
        return endpoint

    def unregister(self, endpoint_id: str) -> bool:
        """Remove a webhook endpoint."""
        with self._lock:
            return self._endpoints.pop(endpoint_id, None) is not None

    def get_endpoint(self, endpoint_id: str) -> WebhookEndpoint | None:
        """Get a specific endpoint."""
        with self._lock:
            return self._endpoints.get(endpoint_id)

    def list_endpoints(self) -> list[WebhookEndpoint]:
        """List all registered endpoints."""
        with self._lock:
            return list(self._endpoints.values())

    def build_payload(
        self, event: WebhookEvent, data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a webhook payload with standard envelope."""
        return {
            "id": str(uuid.uuid4())[:12],
            "event": event.value,
            "timestamp": time.time(),
            "data": data,
        }

    def dispatch(
        self,
        event: WebhookEvent,
        data: dict[str, Any],
    ) -> list[WebhookDelivery]:
        """Dispatch an event to all matching endpoints.

        Returns delivery records. Actual HTTP calls should be done async
        by the caller — this method prepares the deliveries.
        """
        payload = self.build_payload(event, data)
        deliveries = []

        with self._lock:
            self._event_counts[event.value] = (
                self._event_counts.get(event.value, 0) + 1
            )

            for endpoint in self._endpoints.values():
                if not endpoint.active:
                    continue
                if event not in endpoint.events:
                    continue
                if endpoint.failure_count >= self.MAX_FAILURES:
                    endpoint.active = False
                    continue

                delivery = WebhookDelivery(
                    endpoint_id=endpoint.id,
                    event=event.value,
                    payload=payload,
                )
                deliveries.append(delivery)
                endpoint.last_triggered = time.time()

            self._deliveries.extend(deliveries)
            # Trim old deliveries
            if len(self._deliveries) > self.MAX_DELIVERIES:
                self._deliveries = self._deliveries[-self.MAX_DELIVERIES:]

        return deliveries

    def record_result(
        self,
        delivery_id: str,
        status_code: int,
        success: bool,
        error: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """Record the result of a webhook delivery attempt."""
        with self._lock:
            for d in reversed(self._deliveries):
                if d.id == delivery_id:
                    d.status_code = status_code
                    d.success = success
                    d.error = error
                    d.duration_ms = duration_ms
                    break

            # Update endpoint failure count
            for d in reversed(self._deliveries):
                if d.id == delivery_id:
                    ep = self._endpoints.get(d.endpoint_id)
                    if ep:
                        if success:
                            ep.failure_count = 0
                        else:
                            ep.failure_count += 1
                    break

    def get_deliveries(
        self, endpoint_id: str | None = None, limit: int = 50,
    ) -> list[WebhookDelivery]:
        """Get recent delivery records."""
        with self._lock:
            results = []
            for d in reversed(self._deliveries):
                if endpoint_id and d.endpoint_id != endpoint_id:
                    continue
                results.append(d)
                if len(results) >= limit:
                    break
            return results

    def get_stats(self) -> dict:
        """Get webhook system statistics."""
        with self._lock:
            total = len(self._deliveries)
            success = sum(1 for d in self._deliveries if d.success)
            return {
                "total_endpoints": len(self._endpoints),
                "active_endpoints": sum(
                    1 for e in self._endpoints.values() if e.active
                ),
                "total_deliveries": total,
                "successful_deliveries": success,
                "success_rate": round(success / total, 4) if total > 0 else 0.0,
                "event_counts": dict(self._event_counts),
            }
