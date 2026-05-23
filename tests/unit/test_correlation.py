"""Tests for correlation ID middleware."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from llmstack.gateway.middleware.correlation import (
    CorrelationMiddleware,
    CORRELATION_HEADER,
    get_correlation_id,
)


def _create_app():
    app = Starlette()

    @app.route("/test")
    async def test_route(request: Request):
        cid = get_correlation_id(request)
        return PlainTextResponse(cid)

    app.add_middleware(CorrelationMiddleware)
    return app


@pytest.fixture
def client():
    return TestClient(_create_app())


class TestCorrelationMiddleware:
    def test_generates_id_when_missing(self, client):
        response = client.get("/test")
        assert response.status_code == 200
        assert CORRELATION_HEADER in response.headers
        assert len(response.headers[CORRELATION_HEADER]) > 0

    def test_propagates_existing_id(self, client):
        response = client.get("/test", headers={CORRELATION_HEADER: "my-trace-123"})
        assert response.headers[CORRELATION_HEADER] == "my-trace-123"
        assert response.text == "my-trace-123"

    def test_unique_ids_per_request(self, client):
        r1 = client.get("/test")
        r2 = client.get("/test")
        assert r1.headers[CORRELATION_HEADER] != r2.headers[CORRELATION_HEADER]

    def test_id_in_response_body(self, client):
        response = client.get("/test", headers={CORRELATION_HEADER: "test-id"})
        assert response.text == "test-id"

    def test_generated_id_is_uuid_format(self, client):
        response = client.get("/test")
        cid = response.headers[CORRELATION_HEADER]
        # UUID format: 8-4-4-4-12
        parts = cid.split("-")
        assert len(parts) == 5

    def test_get_correlation_id_fallback(self):
        # Without middleware, should return "unknown"
        from starlette.requests import Request

        scope = {"type": "http", "headers": [], "method": "GET", "path": "/"}
        request = Request(scope)
        assert get_correlation_id(request) == "unknown"
