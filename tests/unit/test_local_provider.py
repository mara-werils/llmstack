"""Tests for the local (Ollama/vLLM) provider."""

from __future__ import annotations

import httpx
import pytest

from llmstack.gateway.providers.base import ProviderError
from llmstack.gateway.providers.local import LocalProvider

CHAT_OK = {
    "model": "llama3",
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}


class _Resp:
    def __init__(self, status_code=200, payload=None, *, raise_exc=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._payload


class _StreamCtx:
    def __init__(self, resp, chunks):
        self._resp = resp
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        self._resp.raise_for_status()

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _FakeClient:
    def __init__(self, *, post=None, get=None, post_exc=None, stream_chunks=None,
                 stream_resp=None):
        self._post = post
        self._get = get
        self._post_exc = post_exc
        self._stream_chunks = stream_chunks or []
        self._stream_resp = stream_resp or _Resp()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        if self._post_exc:
            raise self._post_exc
        return self._post

    async def get(self, url):
        return self._get

    def stream(self, method, url, json=None):
        return _StreamCtx(self._stream_resp, self._stream_chunks)


def _patch(monkeypatch, **kw):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(**kw))


def test_default_base_url_appends_v1():
    p = LocalProvider()
    assert p.base_url.endswith("/v1")


def test_explicit_base_url_used():
    p = LocalProvider(base_url="http://host:1234/v1")
    assert p.base_url == "http://host:1234/v1"


async def test_chat_success(monkeypatch):
    _patch(monkeypatch, post=_Resp(200, CHAT_OK))
    resp = await LocalProvider().chat({"model": "llama3"})
    assert resp.content == "hello"
    assert resp.provider == "local"
    assert resp.input_tokens == 7
    assert resp.output_tokens == 3
    assert resp.cost_usd == 0.0
    assert resp.latency_ms >= 0


async def test_chat_http_status_error_5xx_retryable(monkeypatch):
    err = httpx.HTTPStatusError("err", request=None, response=httpx.Response(503))
    _patch(monkeypatch, post=_Resp(raise_exc=err))
    with pytest.raises(ProviderError) as ei:
        await LocalProvider().chat({"model": "x"})
    assert ei.value.status_code == 503
    assert ei.value.retryable is True


async def test_chat_http_status_error_4xx_not_retryable(monkeypatch):
    err = httpx.HTTPStatusError("err", request=None, response=httpx.Response(400))
    _patch(monkeypatch, post=_Resp(raise_exc=err))
    with pytest.raises(ProviderError) as ei:
        await LocalProvider().chat({"model": "x"})
    assert ei.value.retryable is False


async def test_chat_connect_error_retryable(monkeypatch):
    _patch(monkeypatch, post_exc=httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as ei:
        await LocalProvider().chat({"model": "x"})
    assert ei.value.retryable is True


async def test_chat_stream_yields_chunks(monkeypatch):
    _patch(monkeypatch, stream_chunks=[b"a", b"b"])
    out = [c async for c in LocalProvider().chat_stream({"model": "x"})]
    assert out == [b"a", b"b"]


async def test_chat_stream_connect_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ReadTimeout("slow")

    fake = _FakeClient()
    fake.stream = boom
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)
    with pytest.raises(ProviderError) as ei:
        async for _ in LocalProvider().chat_stream({"model": "x"}):
            pass
    assert ei.value.retryable is True


async def test_list_models_success(monkeypatch):
    payload = {"data": [{"id": "llama3", "context_length": 4096}, {"id": "qwen"}]}
    _patch(monkeypatch, get=_Resp(200, payload))
    models = await LocalProvider().list_models()
    assert [m.id for m in models] == ["llama3", "qwen"]
    assert models[0].context_length == 4096
    assert models[1].context_length == 8192  # default


async def test_list_models_falls_back_on_error(monkeypatch):
    p = LocalProvider()
    p._models = []
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: _FakeClient(get=_Resp(raise_exc=ValueError("x")))
    )
    assert await p.list_models() == []


async def test_health_check_true_when_models_listed(monkeypatch):
    _patch(monkeypatch, get=_Resp(200, {"data": []}))
    assert await LocalProvider().health_check() is True
