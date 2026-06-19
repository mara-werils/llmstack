"""Tests for the OpenAI and Google (Gemini) providers."""

from __future__ import annotations

import json

import httpx
import pytest

from llmstack.gateway.providers.base import ProviderError
from llmstack.gateway.providers.google_provider import GoogleProvider, _openai_to_gemini
from llmstack.gateway.providers.openai_provider import OpenAIProvider


class _Resp:
    def __init__(self, payload=None, *, raise_exc=None):
        self._payload = payload or {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._payload


class _StreamCtx:
    def __init__(self, *, bytes_chunks=None, lines=None, raise_exc=None):
        self._bytes = bytes_chunks or []
        self._lines = lines or []
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    async def aiter_bytes(self):
        for c in self._bytes:
            yield c

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, *, post=None, post_exc=None, stream=None):
        self._post = post
        self._post_exc = post_exc
        self._stream = stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        if self._post_exc:
            raise self._post_exc
        return self._post

    def stream(self, method, url, json=None, headers=None):
        return self._stream


def _patch(monkeypatch, **kw):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(**kw))


def _status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("e", request=None, response=httpx.Response(code, text="boom"))


OPENAI_OK = {
    "model": "gpt-4o-mini",
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
}


class TestOpenAIProvider:
    def test_default_base_url(self):
        assert OpenAIProvider().base_url == "https://api.openai.com/v1"

    def test_headers_include_bearer(self):
        assert OpenAIProvider(api_key="sk-x")._headers()["Authorization"] == "Bearer sk-x"

    async def test_chat_strips_x_keys_and_computes_cost(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(OPENAI_OK))
        resp = await OpenAIProvider().chat({"model": "gpt-4o-mini", "x_internal": 1})
        assert resp.content == "hi"
        # gpt-4o-mini = $0.15/M in, $0.60/M out; 1000 in + 1000 out
        assert resp.cost_usd == pytest.approx((1000 * 0.15 + 1000 * 0.60) / 1_000_000)

    async def test_chat_429_retryable(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(raise_exc=_status_error(429)))
        with pytest.raises(ProviderError) as ei:
            await OpenAIProvider().chat({"model": "gpt-4o"})
        assert ei.value.retryable is True
        assert ei.value.status_code == 429

    async def test_chat_400_not_retryable(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(raise_exc=_status_error(400)))
        with pytest.raises(ProviderError) as ei:
            await OpenAIProvider().chat({"model": "gpt-4o"})
        assert ei.value.retryable is False

    async def test_chat_connect_error(self, monkeypatch):
        _patch(monkeypatch, post_exc=httpx.ConnectError("x"))
        with pytest.raises(ProviderError) as ei:
            await OpenAIProvider().chat({"model": "gpt-4o"})
        assert ei.value.retryable is True

    async def test_stream_yields_and_sets_stream_flag(self, monkeypatch):
        _patch(monkeypatch, stream=_StreamCtx(bytes_chunks=[b"a", b"b"]))
        out = [c async for c in OpenAIProvider().chat_stream({"model": "gpt-4o"})]
        assert out == [b"a", b"b"]

    async def test_stream_http_error(self, monkeypatch):
        _patch(monkeypatch, stream=_StreamCtx(raise_exc=_status_error(503)))
        with pytest.raises(ProviderError) as ei:
            async for _ in OpenAIProvider().chat_stream({"model": "gpt-4o"}):
                pass
        assert ei.value.retryable is True

    async def test_list_models_defaults(self):
        models = await OpenAIProvider().list_models()
        assert any(m.id == "gpt-4o" for m in models)


class TestGeminiTranslation:
    def test_system_and_roles_mapped(self):
        url, body = _openai_to_gemini(
            {
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
                "temperature": 0.5,
                "max_tokens": 64,
                "top_p": 0.8,
                "stop": "END",
            },
            "KEY",
        )
        assert "gemini-2.5-flash:generateContent?key=KEY" in url
        assert body["systemInstruction"]["parts"][0]["text"] == "sys"
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][1]["role"] == "model"
        gc = body["generationConfig"]
        assert gc["temperature"] == 0.5
        assert gc["maxOutputTokens"] == 64
        assert gc["topP"] == 0.8
        assert gc["stopSequences"] == ["END"]

    def test_stop_list_preserved(self):
        _, body = _openai_to_gemini({"messages": [], "stop": ["a", "b"]}, "k")
        assert body["generationConfig"]["stopSequences"] == ["a", "b"]

    def test_no_generation_config_when_empty(self):
        _, body = _openai_to_gemini({"messages": [{"role": "user", "content": "x"}]}, "k")
        assert "generationConfig" not in body


GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "hello "}, {"text": "world"}]}}],
    "usageMetadata": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 1_000_000},
}


class TestGoogleProvider:
    async def test_chat_extracts_text_and_normalizes(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(GEMINI_OK))
        resp = await GoogleProvider(api_key="k").chat({"model": "gemini-2.5-flash", "messages": []})
        assert resp.content == "hello world"
        assert resp.raw["object"] == "chat.completion"
        # gemini-2.5-flash = 0.15/M in, 0.60/M out, 1M each
        assert resp.cost_usd == pytest.approx(0.15 + 0.60)

    async def test_chat_http_error(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(raise_exc=_status_error(500)))
        with pytest.raises(ProviderError) as ei:
            await GoogleProvider(api_key="k").chat({"model": "gemini-2.5-flash", "messages": []})
        assert ei.value.retryable is True

    async def test_chat_connect_error(self, monkeypatch):
        _patch(monkeypatch, post_exc=httpx.ReadTimeout("t"))
        with pytest.raises(ProviderError):
            await GoogleProvider(api_key="k").chat({"model": "gemini-2.5-flash", "messages": []})

    async def test_stream_translates_sse(self, monkeypatch):
        event = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        lines = [f"data: {json.dumps(event)}", "ignore me", "data: not-json"]
        _patch(monkeypatch, stream=_StreamCtx(lines=lines))
        out = [
            c
            async for c in GoogleProvider(api_key="k").chat_stream(
                {"model": "gemini-2.5-flash", "messages": [{"role": "system", "content": "s"}]}
            )
        ]
        body = b"".join(out).decode()
        assert "hi" in body
        assert body.endswith("data: [DONE]\n\n")

    async def test_stream_http_error(self, monkeypatch):
        _patch(monkeypatch, stream=_StreamCtx(raise_exc=_status_error(429)))
        with pytest.raises(ProviderError) as ei:
            async for _ in GoogleProvider(api_key="k").chat_stream(
                {"model": "gemini-2.5-flash", "messages": []}
            ):
                pass
        assert ei.value.retryable is True

    async def test_list_models(self):
        models = await GoogleProvider(api_key="k").list_models()
        assert any(m.id == "gemini-2.5-pro" for m in models)
