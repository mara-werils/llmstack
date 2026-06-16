"""Tests for the RAG pipeline (retrieve -> augment -> generate).

Drives RAGPipeline.query (non-stream) and query_stream (async generator),
covering retrieval / no-results branches and error paths. The vector store
and the inference HTTP backend are mocked — no real network is used.
"""

from __future__ import annotations

import httpx
import pytest

from llmstack.gateway.rag import pipeline as pipeline_mod
from llmstack.gateway.rag.pipeline import (
    RAGPipeline,
    RAGResponse,
    RAGStreamChunk,
)
from llmstack.gateway.rag.store import SearchResult

INFERENCE_URL = "http://inference:8080/v1"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeStore:
    """Stand-in for VectorStore.search."""

    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error
        self.calls = []

    async def search(self, query, top_k=5, score_threshold=0.3):
        self.calls.append({"query": query, "top_k": top_k, "score_threshold": score_threshold})
        if self._error is not None:
            raise self._error
        return self._results


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.raised = False

    def raise_for_status(self):
        if self.status_code >= 400:
            self.raised = True
            raise httpx.HTTPStatusError(
                "err", request=None, response=httpx.Response(self.status_code)
            )

    def json(self):
        return self._payload


class _PostClient:
    """Fake AsyncClient supporting only .post (used by query)."""

    last_json = None

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        _PostClient.last_json = json
        self.url = url
        return self._resp


class _StreamCtx:
    """Async context manager returned by client.stream(...)."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=None, response=httpx.Response(self.status_code)
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamClient:
    """Fake AsyncClient supporting .stream (used by query_stream)."""

    last_json = None

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self._status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, json=None):
        _StreamClient.last_json = json
        self.method = method
        self.url = url
        return _StreamCtx(self._lines, status_code=self._status_code)


def _sr(text="chunk text", score=0.91, source="doc.txt"):
    return SearchResult(text=text, score=score, metadata={"source": source})


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: store)


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)


@pytest.fixture
def pipe():
    return RAGPipeline(inference_url=INFERENCE_URL)


# --------------------------------------------------------------------------- #
# query (non-stream)
# --------------------------------------------------------------------------- #
class TestQuery:
    async def test_returns_answer_and_sources(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr(text="x" * 500, score=0.876543, source="a.txt")])
        _patch_store(monkeypatch, store)
        completion = {
            "choices": [{"message": {"content": "The answer."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
        _patch_client(monkeypatch, _PostClient(_Resp(200, completion)))

        resp = await pipe.query("what is x?", model="llama3.2", top_k=3, score_threshold=0.5)

        assert isinstance(resp, RAGResponse)
        assert resp.answer == "The answer."
        assert resp.model == "llama3.2"
        assert resp.usage["total_tokens"] == 14
        # source text truncated to 200 chars, score rounded to 4 dp
        assert len(resp.sources) == 1
        assert resp.sources[0]["source"] == "a.txt"
        assert resp.sources[0]["score"] == 0.8765
        assert len(resp.sources[0]["text"]) == 200
        # latency timings recorded
        assert "retrieval_ms" in resp.latency
        assert "generation_ms" in resp.latency

    async def test_passes_top_k_and_threshold_to_store(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        completion = {"choices": [{"message": {"content": "hi"}}], "usage": {}}
        _patch_client(monkeypatch, _PostClient(_Resp(200, completion)))

        await pipe.query("q", top_k=7, score_threshold=0.42)

        assert store.calls[0]["top_k"] == 7
        assert store.calls[0]["score_threshold"] == 0.42

    async def test_builds_non_stream_request_payload(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr(text="ctx", source="src.md")])
        _patch_store(monkeypatch, store)
        completion = {"choices": [{"message": {"content": "ans"}}], "usage": {}}
        _patch_client(monkeypatch, _PostClient(_Resp(200, completion)))

        await pipe.query("question?", model="m1", temperature=0.7, max_tokens=256)

        sent = _PostClient.last_json
        assert sent["model"] == "m1"
        assert sent["stream"] is False
        assert sent["temperature"] == 0.7
        assert sent["max_tokens"] == 256
        # system prompt carries the context; user message carries the question
        assert sent["messages"][0]["role"] == "system"
        assert "ctx" in sent["messages"][0]["content"]
        assert "src.md" in sent["messages"][0]["content"]
        assert sent["messages"][1] == {"role": "user", "content": "question?"}

    async def test_usage_defaults_to_empty_when_missing(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        completion = {"choices": [{"message": {"content": "ans"}}]}  # no usage key
        _patch_client(monkeypatch, _PostClient(_Resp(200, completion)))

        resp = await pipe.query("q")
        assert resp.usage == {}

    async def test_no_results_returns_canned_response(self, pipe, monkeypatch):
        store = _FakeStore(results=[])
        _patch_store(monkeypatch, store)
        # client should never be used; if it is, this would explode
        _patch_client(monkeypatch, _PostClient(_Resp(500, {})))

        resp = await pipe.query("q", model="my-model")

        assert isinstance(resp, RAGResponse)
        assert "No relevant documents found" in resp.answer
        assert resp.sources == []
        assert resp.model == "my-model"
        assert resp.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        assert "retrieval_ms" in resp.latency
        assert "generation_ms" not in resp.latency

    async def test_http_error_propagates(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        _patch_client(monkeypatch, _PostClient(_Resp(503, {})))

        with pytest.raises(httpx.HTTPStatusError):
            await pipe.query("q")

    async def test_store_error_propagates(self, pipe, monkeypatch):
        store = _FakeStore(error=RuntimeError("store down"))
        _patch_store(monkeypatch, store)
        _patch_client(monkeypatch, _PostClient(_Resp(200, {})))

        with pytest.raises(RuntimeError, match="store down"):
            await pipe.query("q")


# --------------------------------------------------------------------------- #
# query_stream (async generator)
# --------------------------------------------------------------------------- #
async def _collect(agen):
    return [c async for c in agen]


class TestQueryStream:
    async def test_streams_tokens_then_done(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr(text="ctx", score=0.5, source="a.txt")])
        _patch_store(monkeypatch, store)
        lines = [
            'data: {"choices": [{"delta": {"content": "Hel"}}]}',
            'data: {"choices": [{"delta": {"content": "lo"}}]}',
            "data: [DONE]",
        ]
        _patch_client(monkeypatch, _StreamClient(lines))

        chunks = await _collect(pipe.query_stream("q"))

        assert [c.token for c in chunks if c.token] == ["Hel", "lo"]
        # last chunk is the terminal done marker carrying sources
        last = chunks[-1]
        assert isinstance(last, RAGStreamChunk)
        assert last.done is True
        assert last.token == ""
        assert last.sources == [{"text": "ctx", "source": "a.txt", "score": 0.5}]

    async def test_stream_request_payload_has_stream_true(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        _patch_client(monkeypatch, _StreamClient(["data: [DONE]"]))

        await _collect(pipe.query_stream("q", model="mm", temperature=0.2, max_tokens=64))

        sent = _StreamClient.last_json
        assert sent["stream"] is True
        assert sent["model"] == "mm"
        assert sent["temperature"] == 0.2
        assert sent["max_tokens"] == 64

    async def test_no_results_yields_single_done_chunk(self, pipe, monkeypatch):
        store = _FakeStore(results=[])
        _patch_store(monkeypatch, store)
        _patch_client(monkeypatch, _StreamClient(["data: should-not-be-read"]))

        chunks = await _collect(pipe.query_stream("q"))

        assert len(chunks) == 1
        assert chunks[0].token == "No relevant documents found."
        assert chunks[0].done is True
        assert chunks[0].sources == []

    async def test_skips_non_data_lines(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        lines = [
            "",
            ": keep-alive comment",
            "event: message",
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]
        _patch_client(monkeypatch, _StreamClient(lines))

        chunks = await _collect(pipe.query_stream("q"))
        assert [c.token for c in chunks if c.token] == ["ok"]

    async def test_skips_empty_content_deltas(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        lines = [
            'data: {"choices": [{"delta": {"role": "assistant"}}]}',  # no content
            'data: {"choices": [{"delta": {"content": ""}}]}',  # empty content
            'data: {"choices": [{"delta": {"content": "real"}}]}',
            "data: [DONE]",
        ]
        _patch_client(monkeypatch, _StreamClient(lines))

        chunks = await _collect(pipe.query_stream("q"))
        assert [c.token for c in chunks if c.token] == ["real"]

    async def test_tolerates_malformed_json_lines(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        lines = [
            "data: {not valid json",  # JSONDecodeError
            "data: {}",  # KeyError -> choices missing
            'data: {"choices": []}',  # IndexError
            'data: {"choices": [{"delta": {"content": "good"}}]}',
            "data: [DONE]",
        ]
        _patch_client(monkeypatch, _StreamClient(lines))

        chunks = await _collect(pipe.query_stream("q"))
        assert [c.token for c in chunks if c.token] == ["good"]
        assert chunks[-1].done is True

    async def test_stream_falls_through_to_done_without_done_marker(self, pipe, monkeypatch):
        # No "[DONE]" line -> loop ends, trailing terminal chunk is emitted.
        store = _FakeStore(results=[_sr(source="z.txt")])
        _patch_store(monkeypatch, store)
        lines = ['data: {"choices": [{"delta": {"content": "tail"}}]}']
        _patch_client(monkeypatch, _StreamClient(lines))

        chunks = await _collect(pipe.query_stream("q"))

        assert chunks[0].token == "tail"
        assert chunks[-1].done is True
        assert chunks[-1].token == ""
        assert chunks[-1].sources[0]["source"] == "z.txt"

    async def test_stream_http_error_propagates(self, pipe, monkeypatch):
        store = _FakeStore(results=[_sr()])
        _patch_store(monkeypatch, store)
        _patch_client(monkeypatch, _StreamClient(["data: [DONE]"], status_code=500))

        with pytest.raises(httpx.HTTPStatusError):
            await _collect(pipe.query_stream("q"))


# --------------------------------------------------------------------------- #
# _build_context helper
# --------------------------------------------------------------------------- #
class TestBuildContext:
    def test_formats_numbered_sources(self):
        results = [
            _sr(text="first", score=0.9, source="a.txt"),
            _sr(text="second", score=0.8, source="b.txt"),
        ]
        ctx = RAGPipeline._build_context(results)
        assert "[1] (source: a.txt, relevance: 0.90)" in ctx
        assert "[2] (source: b.txt, relevance: 0.80)" in ctx
        assert "first" in ctx
        assert "second" in ctx
        assert "\n\n---\n\n" in ctx

    def test_unknown_source_when_missing(self):
        ctx = RAGPipeline._build_context([SearchResult(text="t", score=0.5, metadata={})])
        assert "source: unknown" in ctx
