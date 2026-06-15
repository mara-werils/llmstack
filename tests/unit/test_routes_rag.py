"""Tests for the /v1/rag API routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.routes import rag as rag_route


class _FakeStore:
    def __init__(self):
        self.deleted = []

    async def ingest(self, text, source, metadata):
        return len(text.split())

    async def delete_by_source(self, source):
        self.deleted.append(source)
        return 1

    async def collection_info(self):
        return {"status": "green", "points_count": 7}


class _Result:
    answer = "the answer"
    sources = [{"source": "a.txt"}]
    model = "llama3.2"
    usage = {"prompt_tokens": 5}
    latency = 12.3


class _Chunk:
    def __init__(self, token="", done=False, sources=None):
        self.token = token
        self.done = done
        self.sources = sources


class _FakePipeline:
    def __init__(self, **kwargs):
        pass

    async def query(self, **kwargs):
        return _Result()

    async def query_stream(self, **kwargs):
        yield _Chunk(token="hel")
        yield _Chunk(token="lo")
        yield _Chunk(done=True, sources=[{"source": "a.txt"}])


@pytest.fixture
def client(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(rag_route, "get_store", lambda: store)
    monkeypatch.setattr(rag_route, "RAGPipeline", _FakePipeline)
    app = FastAPI()
    app.include_router(rag_route.router, prefix="/v1")
    return TestClient(app), store


class TestIngest:
    def test_success(self, client):
        c, _ = client
        resp = c.post("/v1/rag/ingest", json={"text": "hello world", "source": "a.txt"})
        assert resp.status_code == 200
        assert resp.json()["chunks_stored"] == 2

    def test_missing_text_400(self, client):
        c, _ = client
        resp = c.post("/v1/rag/ingest", json={"source": "a.txt"})
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "validation_error"

    def test_too_large_400(self, client):
        c, _ = client
        resp = c.post("/v1/rag/ingest", json={"text": "x" * 1_000_001})
        assert resp.status_code == 400


class TestQuery:
    def test_success(self, client):
        c, _ = client
        resp = c.post("/v1/rag/query", json={"question": "what?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "the answer"
        assert body["sources"] == [{"source": "a.txt"}]
        assert body["latency"] == 12.3

    def test_missing_question_400(self, client):
        c, _ = client
        assert c.post("/v1/rag/query", json={}).status_code == 400

    def test_stream(self, client):
        c, _ = client
        with c.stream("POST", "/v1/rag/query", json={"question": "q", "stream": True}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "hel" in body
        assert '"done": true' in body


class TestDeleteAndStatus:
    def test_delete_document(self, client):
        c, store = client
        resp = c.delete("/v1/rag/documents/a.txt")
        assert resp.status_code == 200
        assert store.deleted == ["a.txt"]

    def test_status(self, client):
        c, _ = client
        assert c.get("/v1/rag/status").json()["points_count"] == 7
