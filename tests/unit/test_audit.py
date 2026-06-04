"""Tests for audit logging."""

from __future__ import annotations
from llmstack.gateway.audit import AuditLogger


class TestAuditLogger:
    def test_singleton(self):
        a1 = AuditLogger.get_instance()
        a2 = AuditLogger.get_instance()
        assert a1 is a2

    def test_log_auth_failure(self):
        al = AuditLogger.get_instance()
        al.log_auth_failure(client_ip="1.2.3.4", reason="invalid key")
        events = al.get_recent_events(limit=1)
        assert len(events) >= 1
        assert events[0]["event_type"] == "auth_failure"

    def test_log_rate_limit(self):
        al = AuditLogger.get_instance()
        al.log_rate_limit(client_ip="1.2.3.4", key_identity="abc")
        events = al.get_recent_events(limit=1)
        assert events[0]["event_type"] == "rate_limit"

    def test_recent_events_limit(self):
        al = AuditLogger.get_instance()
        for i in range(5):
            al.log_auth_failure(client_ip=f"1.2.3.{i}")
        events = al.get_recent_events(limit=3)
        assert len(events) == 3
