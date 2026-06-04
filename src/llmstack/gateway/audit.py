"""Structured audit logging for security events."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any

_logger = logging.getLogger("llmstack.audit")

_MAX_RING_SIZE = 1000


class AuditLogger:
    """Singleton audit logger that emits structured JSON and keeps a ring buffer."""

    _instance: AuditLogger | None = None
    _lock = threading.Lock()

    def __new__(cls) -> AuditLogger:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._ring: deque[dict[str, Any]] = deque(
                        maxlen=_MAX_RING_SIZE,
                    )
                    cls._instance = inst
        return cls._instance

    @classmethod
    def get_instance(cls) -> AuditLogger:
        """Return the singleton instance (creates one if needed)."""
        return cls()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_auth_failure(
        self,
        client_ip: str,
        api_key_hash: str = "",
        details: str = "",
        reason: str = "",
    ) -> None:
        self._emit(
            event_type="auth_failure",
            client_ip=client_ip,
            api_key_hash=api_key_hash[:8],
            action="authenticate",
            outcome="denied",
            details=details or reason,
        )

    def log_rate_limit(
        self,
        client_ip: str,
        api_key_hash: str = "",
        details: str = "",
        key_identity: str = "",
    ) -> None:
        self._emit(
            event_type="rate_limit",
            client_ip=client_ip,
            api_key_hash=api_key_hash[:8],
            action="request",
            outcome="throttled",
            details=details,
        )

    def log_guardrail_violation(
        self,
        client_ip: str,
        api_key_hash: str = "",
        details: str = "",
    ) -> None:
        self._emit(
            event_type="guardrail_violation",
            client_ip=client_ip,
            api_key_hash=api_key_hash[:8],
            action="content_filter",
            outcome="blocked",
            details=details,
        )

    def log_admin_action(
        self,
        client_ip: str,
        api_key_hash: str = "",
        action: str = "",
        details: str = "",
    ) -> None:
        self._emit(
            event_type="admin_action",
            client_ip=client_ip,
            api_key_hash=api_key_hash[:8],
            action=action,
            outcome="executed",
            details=details,
        )

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return up to *limit* most-recent audit events (newest first)."""
        items = list(self._ring)
        items.reverse()
        return items[:limit]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, **fields: Any) -> None:
        record: dict[str, Any] = {
            "timestamp": time.time(),
            **fields,
        }
        self._ring.append(record)
        _logger.info(json.dumps(record, default=str))
