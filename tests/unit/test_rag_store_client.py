"""Tests for the RAG VectorStore HTTP/Qdrant client paths.

Chunking and dataclasses are covered in test_rag_store.py — this file drives
the async embed / ensure_collection / ingest / search / delete paths.
"""

from __future__ import annotations

import httpx
import pytest

from llmstack.gateway.rag import store as store_mod
from llmstack.gateway.rag.store import VectorStore


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=None, response=httpx.Response(self.status_code)
            )

    def json(self):
        return self._payload


class _RoutingClient:
    """Fake AsyncClient that dispatches (method, url) to a handler."""

    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        return self._handler("GET", url, None)

    async def post(self, url, json=None):
        return self._handler("POST", url, json)

    async def put(self, url, json=None):
        return self._handler("PUT", url, json)


def _patch(monkeypatch, handler):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _RoutingClient(handler))


def _embedding_response(n=1):
    return _Resp(200, {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(n)]})


@pytest.fixture
def vstore():
    return VectorStore(qdrant_url="http://qdrant:6333", embeddings_url="http://embed:8080")


class TestEmbed:
    async def test_embed_sets_dimension(self, vstore, monkeypatch):
        _patch(monkeypatch, lambda m, u, j: _embedding_response(2))
        out = await vstore._embed(["a", "b"])
        assert len(out) == 2
        assert vstore._dimension == 3


class TestEnsureCollection:
    async def test_creates_when_missing(self, vstore, monkeypatch):
        calls = []

        def handler(method, url, json):
            calls.append((method, url))
            if url.endswith("/embeddings"):
                return _embedding_response(1)
            if method == "GET":
                return _Resp(404)
            return _Resp(200)

        _patch(monkeypatch, handler)
        await vstore.ensure_collection()
        assert any(m == "PUT" for m, _ in calls)

    async def test_skips_when_exists(self, vstore, monkeypatch):
        vstore._dimension = 3
        puts = []

        def handler(method, url, json):
            if method == "PUT":
                puts.append(url)
            return _Resp(200)

        _patch(monkeypatch, handler)
        await vstore.ensure_collection()
        assert puts == []


class TestIngest:
    async def test_ingest_stores_chunks(self, vstore, monkeypatch):
        vstore._dimension = 3

        def handler(method, url, json):
            if url.endswith("/embeddings"):
                return _embedding_response(len(json["input"]))
            return _Resp(200)

        _patch(monkeypatch, handler)
        count = await vstore.ingest("hello world foo bar", source="s.txt", metadata={"tag": "x"})
        assert count >= 1

    async def test_ingest_empty_returns_zero(self, vstore, monkeypatch):
        vstore._dimension = 3
        _patch(monkeypatch, lambda m, u, j: _Resp(200))
        assert await vstore.ingest("   ", source="s") == 0


class TestSearch:
    async def test_search_maps_results(self, vstore, monkeypatch):
        def handler(method, url, json):
            if url.endswith("/embeddings"):
                return _embedding_response(1)
            return _Resp(
                200,
                {"result": [{"score": 0.9, "payload": {"text": "hit", "source": "s.txt"}}]},
            )

        _patch(monkeypatch, handler)
        results = await vstore.search("query", top_k=3)
        assert len(results) == 1
        assert results[0].text == "hit"
        assert results[0].score == 0.9
        assert results[0].metadata == {"source": "s.txt"}


class TestDeleteAndInfo:
    async def test_delete_by_source(self, vstore, monkeypatch):
        _patch(monkeypatch, lambda m, u, j: _Resp(200, {"result": {"status": "ok"}}))
        assert await vstore.delete_by_source("s.txt") == "ok"

    async def test_collection_info_not_found(self, vstore, monkeypatch):
        _patch(monkeypatch, lambda m, u, j: _Resp(404))
        assert (await vstore.collection_info())["status"] == "not_found"

    async def test_collection_info_ok(self, vstore, monkeypatch):
        _patch(
            monkeypatch,
            lambda m, u, j: _Resp(200, {"result": {"status": "green", "points_count": 5}}),
        )
        info = await vstore.collection_info()
        assert info["status"] == "green"
        assert info["points_count"] == 5


def test_get_store_singleton(monkeypatch):
    monkeypatch.setattr(store_mod, "_store", None)
    assert store_mod.get_store() is store_mod.get_store()
