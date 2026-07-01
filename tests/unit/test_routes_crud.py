"""Tests for the /webhooks, /batch, and /conversations CRUD routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.conversations import ConversationStore
from llmstack.gateway.routes import batch as batch_route
from llmstack.gateway.routes import conversations as conv_route
from llmstack.gateway.routes import webhooks as webhooks_route


def _client(module, prefix=""):
    app = FastAPI()
    app.include_router(module.router, prefix=prefix)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# /webhooks
# --------------------------------------------------------------------------- #
@pytest.fixture
def wh_client(monkeypatch):
    from llmstack.gateway.webhooks import WebhookManager

    monkeypatch.setattr(webhooks_route, "_manager", WebhookManager())
    return _client(webhooks_route)


class TestWebhookRoutes:
    def test_list_empty(self, wh_client):
        assert wh_client.get("/webhooks").json() == {"endpoints": []}

    def test_register_and_delete(self, wh_client):
        resp = wh_client.post(
            "/webhooks", json={"url": "http://localhost:9000/hook", "events": ["request.completed"]}
        )
        assert resp.status_code == 201
        endpoint_id = resp.json()["id"]
        assert wh_client.delete(f"/webhooks/{endpoint_id}").json()["deleted"] is True

    def test_register_invalid_event(self, wh_client):
        resp = wh_client.post(
            "/webhooks", json={"url": "http://localhost/h", "events": ["not.a.real.event"]}
        )
        assert resp.status_code == 400

    def test_delete_missing_404(self, wh_client):
        assert wh_client.delete("/webhooks/ghost").status_code == 404

    def test_deliveries_and_stats(self, wh_client):
        assert wh_client.get("/webhooks/x/deliveries").json() == {"deliveries": []}
        assert wh_client.get("/webhooks/stats").status_code == 200


# --------------------------------------------------------------------------- #
# /batch
# --------------------------------------------------------------------------- #
@pytest.fixture
def batch_client(monkeypatch):
    from llmstack.gateway.batch import BatchProcessor

    monkeypatch.setattr(batch_route, "_processor", BatchProcessor())
    return _client(batch_route)


class TestBatchRoutes:
    def test_create_and_get(self, batch_client):
        resp = batch_client.post(
            "/batch/jobs", json={"requests": [{"model": "m", "messages": []}], "concurrency": 1}
        )
        assert resp.status_code == 201
        job_id = resp.json()["id"]
        assert batch_client.get(f"/batch/jobs/{job_id}").status_code == 200

    def test_list(self, batch_client):
        assert "jobs" in batch_client.get("/batch/jobs").json()

    def test_get_missing_404(self, batch_client):
        assert batch_client.get("/batch/jobs/ghost").status_code == 404

    def test_cancel_missing_400(self, batch_client):
        assert batch_client.post("/batch/jobs/ghost/cancel").status_code == 400

    def test_cancel_pending_job_succeeds(self, batch_client):
        job_id = batch_client.post(
            "/batch/jobs", json={"requests": [{"model": "m", "messages": []}]}
        ).json()["id"]
        assert batch_client.post(f"/batch/jobs/{job_id}/cancel").json() == {"cancelled": True}

    def test_create_too_many_400(self, batch_client):
        big = [{"model": "m", "messages": []}] * 5000
        assert batch_client.post("/batch/jobs", json={"requests": big}).status_code == 400

    def test_get_processor_lazily_creates_and_reuses(self, monkeypatch):
        monkeypatch.setattr(batch_route, "_processor", None)
        first = batch_route.get_processor()
        assert batch_route.get_processor() is first


# --------------------------------------------------------------------------- #
# /conversations
# --------------------------------------------------------------------------- #
@pytest.fixture
def conv_client(monkeypatch, tmp_path):
    monkeypatch.setattr(conv_route, "_store", ConversationStore(db_path=tmp_path / "conv.db"))
    return _client(conv_route)


class TestConversationRoutes:
    def test_create_list_get(self, conv_client):
        created = conv_client.post(
            "/conversations", json={"title": "Chat 1", "model": "llama3"}
        ).json()
        cid = created["id"]

        listed = conv_client.get("/conversations").json()["conversations"]
        assert any(c["id"] == cid for c in listed)

        got = conv_client.get(f"/conversations/{cid}").json()
        assert got["title"] == "Chat 1"
        assert got["messages"] == []

    def test_get_missing_404(self, conv_client):
        assert conv_client.get("/conversations/ghost").status_code == 404

    def test_add_message_and_delete(self, conv_client):
        cid = conv_client.post("/conversations", json={"title": "c"}).json()["id"]
        msg = conv_client.post(
            f"/conversations/{cid}/messages", json={"role": "user", "content": "hi"}
        ).json()
        assert msg["content"] == "hi"

        got = conv_client.get(f"/conversations/{cid}").json()
        assert len(got["messages"]) == 1

        assert conv_client.delete(f"/conversations/{cid}").json()["deleted"] is True
        assert conv_client.get(f"/conversations/{cid}").status_code == 404


# --------------------------------------------------------------------------- #
# Lazy singleton getters — tests above inject fixtures directly, bypassing
# get_manager()/get_store()'s own lazy-init branch.
# --------------------------------------------------------------------------- #
class TestLazySingletonGetters:
    def test_webhooks_get_manager_lazily_creates_and_reuses(self, monkeypatch):
        monkeypatch.setattr(webhooks_route, "_manager", None)
        first = webhooks_route.get_manager()
        assert webhooks_route.get_manager() is first

    def test_conversations_get_store_lazily_creates_and_reuses(self, monkeypatch, tmp_path):
        from llmstack.gateway import conversations as conversations_mod

        monkeypatch.setattr(conversations_mod, "DEFAULT_DB_PATH", tmp_path / "conv.db")
        monkeypatch.setattr(conv_route, "_store", None)
        first = conv_route.get_store()
        assert conv_route.get_store() is first

    def test_add_message_missing_conversation_404(self, conv_client):
        resp = conv_client.post(
            "/conversations/ghost/messages", json={"role": "user", "content": "x"}
        )
        assert resp.status_code == 404

    def test_delete_missing_404(self, conv_client):
        assert conv_client.delete("/conversations/ghost").status_code == 404

    def test_stats(self, conv_client):
        assert conv_client.get("/conversations/stats").status_code == 200
