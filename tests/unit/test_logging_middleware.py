"""Tests for the gateway structured-logging middleware."""

from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.middleware import logging as logging_mod
from llmstack.gateway.middleware.logging import LoggingMiddleware


def _app():
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/v1/models")
    async def models():
        return {"ok": True}

    return app


class TestLoggingMiddleware:
    def test_adds_request_id_header(self):
        client = TestClient(_app())
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers

    def test_uses_forwarded_for_client_ip(self):
        """The first hop in X-Forwarded-For should be used as the client IP
        (see gateway/middleware/logging.py:114)."""
        client = TestClient(_app())
        resp = client.get("/v1/models", headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"})
        assert resp.status_code == 200


class TestSetupLogger:
    def test_text_format_uses_plain_formatter(self, monkeypatch):
        """LOG_FORMAT=text takes the non-JSON branch in _setup_logger
        (gateway/middleware/logging.py:35)."""
        saved_handlers = list(logging_mod.logger.handlers)
        monkeypatch.setattr(logging_mod, "LOG_FORMAT", "text")
        try:
            logging_mod._setup_logger()
            added = logging_mod.logger.handlers[-1]
            assert not isinstance(added.formatter, logging_mod._JsonFormatter)
        finally:
            logging_mod.logger.handlers = saved_handlers
