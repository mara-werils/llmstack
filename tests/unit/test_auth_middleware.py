"""Tests for the gateway API-key auth middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.middleware.auth import AuthMiddleware, _hash_key


def _app(api_keys):
    app = FastAPI()
    app.add_middleware(AuthMiddleware, api_keys=api_keys)

    @app.get("/v1/models")
    async def models():
        return {"ok": True}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    @app.get("/healthz/ready")
    async def health_ready():
        return {"ok": True}

    @app.get("/healthz/live")
    async def health_live():
        return {"ok": True}

    @app.get("/ui/index")
    async def ui():
        return {"ok": True}

    return app


def test_hash_key_is_sha256():
    assert _hash_key("secret") == _hash_key("secret")
    assert _hash_key("a") != _hash_key("b")
    assert len(_hash_key("x")) == 64


def test_blank_keys_filtered():
    mw = AuthMiddleware(app=None, api_keys=["  ", "", "real-key"])
    assert len(mw._key_map) == 1
    assert next(iter(mw._key_map.values())) == "real-key"[:8]


class TestDispatch:
    def test_no_keys_means_open(self):
        client = TestClient(_app([]))
        assert client.get("/v1/models").status_code == 200

    def test_valid_key_allowed(self):
        client = TestClient(_app(["sekret-123"]))
        resp = client.get("/v1/models", headers={"Authorization": "Bearer sekret-123"})
        assert resp.status_code == 200

    def test_invalid_key_rejected(self):
        client = TestClient(_app(["sekret-123"]))
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "auth_error"

    def test_missing_header_rejected(self):
        client = TestClient(_app(["sekret-123"]))
        assert client.get("/v1/models").status_code == 401

    def test_non_bearer_header_rejected(self):
        client = TestClient(_app(["sekret-123"]))
        resp = client.get("/v1/models", headers={"Authorization": "Basic abc"})
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "path", ["/healthz", "/healthz/ready", "/healthz/live", "/ui/index"]
    )
    def test_skip_paths_bypass_auth(self, path):
        client = TestClient(_app(["sekret-123"]))
        assert client.get(path).status_code == 200
