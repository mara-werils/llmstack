"""Tests for AnthropicProvider HTTP chat/stream paths."""

from __future__ import annotations

import json

import httpx
import pytest

from llmstack.gateway.providers.anthropic_provider import (
    AnthropicProvider,
    _openai_to_anthropic,
)
from llmstack.gateway.providers.base import ProviderError


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
    def __init__(self, *, lines=None, raise_exc=None):
        self._lines = lines or []
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

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
    return httpx.HTTPStatusError("e", request=None, response=httpx.Response(code, text="x"))


MESSAGES_OK = {
    "id": "msg_1",
    "model": "claude-sonnet-4-20250514",
    "content": [{"type": "text", "text": "hi "}, {"type": "thinking", "text": "ignore"},
                {"type": "text", "text": "there"}],
    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    "stop_reason": "max_tokens",
}


class TestTranslationEdges:
    def test_assistant_first_gets_user_prepended(self):
        body = _openai_to_anthropic({"messages": [{"role": "assistant", "content": "a"}]})
        assert body["messages"][0]["role"] == "user"

    def test_consecutive_same_role_merged(self):
        body = _openai_to_anthropic(
            {"messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ]}
        )
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"] == "a\nb"

    def test_top_p_passed_through(self):
        body = _openai_to_anthropic({"messages": [{"role": "user", "content": "x"}], "top_p": 0.5})
        assert body["top_p"] == 0.5


class TestChat:
    def test_default_base_url_and_headers(self):
        p = AnthropicProvider(api_key="ak")
        assert p.base_url == "https://api.anthropic.com"
        h = p._headers()
        assert h["x-api-key"] == "ak"
        assert h["anthropic-version"] == "2023-06-01"

    async def test_chat_translates_response(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(MESSAGES_OK))
        resp = await AnthropicProvider(api_key="k").chat(
            {"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "q"}]}
        )
        assert resp.content == "hi there"
        assert resp.raw["choices"][0]["finish_reason"] == "length"
        # sonnet-4 = 3/M in, 15/M out, 1M each
        assert resp.cost_usd == pytest.approx(3.00 + 15.00)

    async def test_chat_529_retryable(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(raise_exc=_status_error(529)))
        with pytest.raises(ProviderError) as ei:
            await AnthropicProvider(api_key="k").chat({"messages": []})
        assert ei.value.retryable is True

    async def test_chat_400_not_retryable(self, monkeypatch):
        _patch(monkeypatch, post=_Resp(raise_exc=_status_error(400)))
        with pytest.raises(ProviderError) as ei:
            await AnthropicProvider(api_key="k").chat({"messages": []})
        assert ei.value.retryable is False

    async def test_chat_connect_error(self, monkeypatch):
        _patch(monkeypatch, post_exc=httpx.ConnectError("x"))
        with pytest.raises(ProviderError) as ei:
            await AnthropicProvider(api_key="k").chat({"messages": []})
        assert ei.value.retryable is True


class TestChatStream:
    async def test_stream_translates_deltas(self, monkeypatch):
        delta = {"type": "content_block_delta", "delta": {"text": "yo"}}
        stop = {"type": "message_stop"}
        lines = [
            f"data: {json.dumps(delta)}",
            "event: ping",
            "data: not-json",
            f"data: {json.dumps(stop)}",
        ]
        _patch(monkeypatch, stream=_StreamCtx(lines=lines))
        out = b"".join([
            c async for c in AnthropicProvider(api_key="k").chat_stream({"model": "m", "messages": []})
        ]).decode()
        assert "yo" in out
        assert out.endswith("data: [DONE]\n\n")

    async def test_stream_done_marker(self, monkeypatch):
        _patch(monkeypatch, stream=_StreamCtx(lines=["data: [DONE]"]))
        out = [c async for c in AnthropicProvider(api_key="k").chat_stream({"messages": []})]
        assert out == [b"data: [DONE]\n\n"]

    async def test_stream_http_error(self, monkeypatch):
        _patch(monkeypatch, stream=_StreamCtx(raise_exc=_status_error(503)))
        with pytest.raises(ProviderError) as ei:
            async for _ in AnthropicProvider(api_key="k").chat_stream({"messages": []}):
                pass
        assert ei.value.retryable is True

    async def test_list_models(self):
        models = await AnthropicProvider(api_key="k").list_models()
        assert any("claude" in m.id for m in models)
