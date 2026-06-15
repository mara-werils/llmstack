"""Tests for the playground API store and routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.routes import playground_api as pg


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(pg.Path, "home", lambda: tmp_path)
    return pg.PlaygroundStore()


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(pg, "_store", store)
    app = FastAPI()
    app.include_router(pg.router)
    return TestClient(app)


class TestStore:
    def test_save_assigns_id_and_timestamps(self, store):
        saved = store.save_session(pg.PlaygroundSession(title="t"))
        assert len(saved.id) == 12
        assert saved.created_at > 0
        assert saved.updated_at >= saved.created_at

    def test_save_preserves_existing_id(self, store):
        s = store.save_session(pg.PlaygroundSession(id="abc", title="t"))
        s.title = "updated"
        again = store.save_session(s)
        assert again.id == "abc"
        assert store.get_session("abc").title == "updated"

    def test_get_missing_returns_none(self, store):
        assert store.get_session("nope") is None

    def test_roundtrip_messages(self, store):
        msgs = [{"role": "user", "content": "hi"}]
        saved = store.save_session(pg.PlaygroundSession(messages=msgs))
        assert store.get_session(saved.id).messages == msgs

    def test_list_orders_by_updated(self, store):
        store.save_session(pg.PlaygroundSession(id="a", title="A"))
        store.save_session(pg.PlaygroundSession(id="b", title="B"))
        listed = store.list_sessions()
        assert {row["id"] for row in listed} == {"a", "b"}
        assert listed[0]["id"] == "b"  # most recently saved first

    def test_list_respects_limit(self, store):
        for i in range(3):
            store.save_session(pg.PlaygroundSession(id=f"s{i}"))
        assert len(store.list_sessions(limit=2)) == 2

    def test_delete(self, store):
        store.save_session(pg.PlaygroundSession(id="x"))
        assert store.delete_session("x") is True
        assert store.delete_session("missing") is False

    def test_share_and_get(self, store):
        store.save_session(pg.PlaygroundSession(id="s", title="orig", model="llama3"))
        share_id = store.share_session("s", title="shared title")
        assert len(share_id) == 16
        shared = store.get_shared(share_id)
        assert shared["share_id"] == share_id
        assert shared["title"] == "shared title"
        assert shared["model"] == "llama3"

    def test_share_uses_session_title_when_blank(self, store):
        store.save_session(pg.PlaygroundSession(id="s", title="orig"))
        shared = store.get_shared(store.share_session("s"))
        assert shared["title"] == "orig"

    def test_share_missing_session_raises(self, store):
        with pytest.raises(ValueError):
            store.share_session("ghost")

    def test_get_shared_missing(self, store):
        assert store.get_shared("nope") is None


class TestRoutes:
    def test_save_then_list_and_get(self, client):
        resp = client.post("/v1/playground/sessions", json={"title": "hello"})
        assert resp.status_code == 200
        sid = resp.json()["id"]

        listed = client.get("/v1/playground/sessions").json()["sessions"]
        assert any(s["id"] == sid for s in listed)

        got = client.get(f"/v1/playground/sessions/{sid}").json()
        assert got["title"] == "hello"

    def test_get_missing_404(self, client):
        assert client.get("/v1/playground/sessions/ghost").status_code == 404

    def test_delete(self, client):
        sid = client.post("/v1/playground/sessions", json={"id": "d1"}).json()["id"]
        assert client.delete(f"/v1/playground/sessions/{sid}").status_code == 200
        assert client.delete(f"/v1/playground/sessions/{sid}").status_code == 404

    def test_share_flow(self, client):
        sid = client.post("/v1/playground/sessions", json={"id": "sh", "title": "x"}).json()["id"]
        share = client.post("/v1/playground/share", json={"session_id": sid}).json()
        assert share["status"] == "shared"
        shared = client.get(f"/v1/playground/shared/{share['share_id']}").json()
        assert shared["title"] == "x"

    def test_share_missing_404(self, client):
        assert client.post("/v1/playground/share", json={"session_id": "ghost"}).status_code == 404

    def test_get_shared_missing_404(self, client):
        assert client.get("/v1/playground/shared/nope").status_code == 404
