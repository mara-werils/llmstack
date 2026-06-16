"""Tests for the /v1/templates API routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.prompt_templates import TemplateStore
from llmstack.gateway.routes import templates as templates_route


@pytest.fixture
def client(monkeypatch):
    """Mount the templates router with a fresh in-memory store per test."""
    store = TemplateStore()
    monkeypatch.setattr(templates_route, "get_store", lambda: store)
    app = FastAPI()
    app.include_router(templates_route.router, prefix="/v1")
    return TestClient(app), store


def _seed(store, name="greet", content="Hello {{name}}!"):
    """Create a template in the store and return it."""
    return store.create(name=name, content=content, description="d", tags=["x"])


# --- get_store singleton (lines 17-25) ---


class TestGetStore:
    def test_singleton_and_builtins_loaded(self, monkeypatch):
        # Reset module-level singleton so get_store builds a fresh one.
        monkeypatch.setattr(templates_route, "_store", None)
        store = templates_route.get_store()
        assert store is templates_route.get_store()  # cached singleton
        # Built-in templates loaded; duplicate create raises ValueError swallowed.
        assert store.get("code-review") is not None
        assert store.count >= 5


# --- list_templates (lines 61-66) ---


class TestList:
    def test_empty(self, client):
        c, _ = client
        resp = c.get("/v1/templates")
        assert resp.status_code == 200
        body = resp.json()
        assert body["templates"] == []
        assert body["total"] == 0

    def test_lists_created(self, client):
        c, store = client
        _seed(store)
        body = c.get("/v1/templates").json()
        assert body["total"] == 1
        assert body["templates"][0]["name"] == "greet"

    def test_filter_by_category(self, client):
        c, store = client
        store.create(name="a", content="x", category="dev")
        store.create(name="b", content="y", category="general")
        body = c.get("/v1/templates", params={"category": "dev"}).json()
        assert [t["name"] for t in body["templates"]] == ["a"]

    def test_filter_by_tag_and_limit(self, client):
        c, store = client
        store.create(name="a", content="x", tags=["keep"])
        store.create(name="b", content="y", tags=["keep"])
        body = c.get("/v1/templates", params={"tag": "keep", "limit": 1}).json()
        assert len(body["templates"]) == 1


# --- create_template (lines 72-83) ---


class TestCreate:
    def test_success(self, client):
        c, _ = client
        resp = c.post(
            "/v1/templates",
            json={
                "name": "greet",
                "content": "Hi {{name}}",
                "description": "desc",
                "category": "general",
                "tags": ["a", "b"],
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "greet"
        assert body["variables"] == ["name"]
        assert body["tags"] == ["a", "b"]

    def test_duplicate_returns_409(self, client):
        c, store = client
        _seed(store)
        resp = c.post("/v1/templates", json={"name": "greet", "content": "X"})
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_missing_required_field_422(self, client):
        c, _ = client
        # No content -> pydantic validation error.
        assert c.post("/v1/templates", json={"name": "x"}).status_code == 422


# --- get_template (lines 89-93) ---


class TestGet:
    def test_by_name(self, client):
        c, store = client
        _seed(store)
        resp = c.get("/v1/templates/greet")
        assert resp.status_code == 200
        assert resp.json()["name"] == "greet"

    def test_by_id(self, client):
        c, store = client
        t = _seed(store)
        assert c.get(f"/v1/templates/{t.id}").json()["id"] == t.id

    def test_not_found_404(self, client):
        c, _ = client
        resp = c.get("/v1/templates/missing")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Template not found"


# --- update_template (lines 99-103) ---


class TestUpdate:
    def test_success_bumps_version(self, client):
        c, store = client
        _seed(store)
        resp = c.put("/v1/templates/greet", json={"content": "Bye {{name}}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_version"] == 2
        assert body["content"] == "Bye {{name}}"

    def test_not_found_404(self, client):
        c, _ = client
        resp = c.put("/v1/templates/missing", json={"content": "x"})
        assert resp.status_code == 404


# --- delete_template (lines 109-112) ---


class TestDelete:
    def test_success(self, client):
        c, store = client
        _seed(store)
        resp = c.delete("/v1/templates/greet")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": True}
        assert store.get("greet") is None

    def test_not_found_404(self, client):
        c, _ = client
        assert c.delete("/v1/templates/missing").status_code == 404


# --- render_template (lines 118-122) ---


class TestRender:
    def test_success(self, client):
        c, store = client
        _seed(store)
        resp = c.post(
            "/v1/templates/greet/render",
            json={"variables": {"name": "World"}},
        )
        assert resp.status_code == 200
        assert resp.json()["rendered"] == "Hello World!"

    def test_empty_variables_returns_raw(self, client):
        c, store = client
        _seed(store)
        resp = c.post("/v1/templates/greet/render", json={})
        assert resp.status_code == 200
        assert resp.json()["rendered"] == "Hello {{name}}!"

    def test_not_found_404(self, client):
        c, _ = client
        resp = c.post("/v1/templates/missing/render", json={"variables": {}})
        assert resp.status_code == 404


# --- rollback_template (lines 128-135) ---


class TestRollback:
    def test_success(self, client):
        c, store = client
        _seed(store)
        store.update("greet", content="v2 {{name}}")  # current_version -> 2
        resp = c.post("/v1/templates/greet/rollback", json={"version": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_version"] == 1
        assert body["content"] == "Hello {{name}}!"

    def test_invalid_version_400(self, client):
        c, store = client
        _seed(store)
        resp = c.post("/v1/templates/greet/rollback", json={"version": 99})
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]

    def test_not_found_404(self, client):
        c, _ = client
        resp = c.post("/v1/templates/missing/rollback", json={"version": 1})
        assert resp.status_code == 404


# --- list_versions (lines 141-158) ---


class TestVersions:
    def test_success(self, client):
        c, store = client
        _seed(store)
        store.update("greet", content="v2 {{name}}")
        resp = c.get("/v1/templates/greet/versions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "greet"
        assert body["current_version"] == 2
        versions = body["versions"]
        assert [v["version"] for v in versions] == [1, 2]
        assert versions[0]["variables"] == ["name"]
        assert "content_preview" in versions[0]

    def test_not_found_404(self, client):
        c, _ = client
        assert c.get("/v1/templates/missing/versions").status_code == 404


# --- search_templates (lines 164-170) ---


class TestSearch:
    def test_match_by_name(self, client):
        c, store = client
        _seed(store, name="greeting")
        store.create(name="other", content="z")
        resp = c.get("/v1/templates/search/greet")
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "greet"
        assert body["total"] == 1
        assert body["results"][0]["name"] == "greeting"

    def test_no_match_empty(self, client):
        c, store = client
        _seed(store)
        body = c.get("/v1/templates/search/zzz").json()
        assert body["total"] == 0
        assert body["results"] == []

    def test_limit_param(self, client):
        c, store = client
        store.create(name="match-1", content="a", tags=["match"])
        store.create(name="match-2", content="b", tags=["match"])
        body = c.get("/v1/templates/search/match", params={"limit": 1}).json()
        assert body["total"] == 1
