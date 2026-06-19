"""Tests for the gateway request-size limit middleware."""

from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.middleware.request_size import RequestSizeMiddleware


def _app(max_bytes=100):
    app = FastAPI()
    app.add_middleware(RequestSizeMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    return app


def test_small_request_allowed():
    client = TestClient(_app())
    assert client.post("/echo", content=b"hi").status_code == 200


def test_oversized_request_rejected():
    client = TestClient(_app(max_bytes=4))
    resp = client.post("/echo", content=b"way too many bytes")
    assert resp.status_code == 413
    assert resp.json()["error"]["type"] == "payload_too_large"


class _FakeURL:
    path = "/echo"


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers
        self.url = _FakeURL()
        self.client = None


async def test_malformed_content_length_does_not_500():
    # A non-numeric Content-Length is attacker-controllable; parsing it must
    # not raise a ValueError (500). Dispatch directly so the bad header reaches
    # the middleware instead of being recomputed by the HTTP client.
    mw = RequestSizeMiddleware(app=None, max_bytes=100)
    passed = {}

    async def call_next(_request):
        passed["hit"] = True
        return "ok"

    result = await mw.dispatch(_FakeRequest({"content-length": "not-a-number"}), call_next)
    assert result == "ok"
    assert passed["hit"]
