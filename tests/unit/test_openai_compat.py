"""Tests for OpenAI-compatible provider adapters: Groq, Together, Mistral.

Each subclass delegates chat() to OpenAIProvider.chat() and then overrides
``resp.provider`` with its own name. httpx is mocked — no real network.
"""

from __future__ import annotations

import httpx
import pytest

from llmstack.gateway.providers.base import ProviderError
from llmstack.gateway.providers.openai_compat import (
    GroqProvider,
    MistralProvider,
    TogetherProvider,
)


class _Resp:
    def __init__(self, payload=None, *, raise_exc=None):
        self._payload = payload or {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, *, post=None, post_exc=None):
        self._post = post
        self._post_exc = post_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        if self._post_exc:
            raise self._post_exc
        return self._post


def _patch(monkeypatch, **kw):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(**kw))


def _status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("e", request=None, response=httpx.Response(code, text="boom"))


OK = {
    "model": "llama-3.3-70b-versatile",
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
}


# --- construction / base_url / models -------------------------------------

def test_groq_base_url_and_name():
    p = GroqProvider(api_key="k")
    assert p.name == "groq"
    assert p.base_url == "https://api.groq.com/openai/v1"


def test_together_base_url_and_name():
    p = TogetherProvider(api_key="k")
    assert p.name == "together"
    assert p.base_url == "https://api.together.xyz/v1"


def test_mistral_base_url_and_name():
    p = MistralProvider(api_key="k")
    assert p.name == "mistral"
    assert p.base_url == "https://api.mistral.ai/v1"


async def test_groq_lists_its_models():
    models = await GroqProvider().list_models()
    ids = {m.id for m in models}
    assert "llama-3.3-70b-versatile" in ids
    assert all(m.provider == "groq" for m in models)


async def test_together_lists_its_models():
    models = await TogetherProvider().list_models()
    assert any(m.id == "deepseek-ai/DeepSeek-R1" for m in models)
    assert all(m.provider == "together" for m in models)


async def test_mistral_lists_its_models():
    models = await MistralProvider().list_models()
    assert any(m.id == "mistral-large-latest" for m in models)
    assert all(m.provider == "mistral" for m in models)


# --- chat() overrides resp.provider (covers lines 66-68/124-126/175-177) ---

async def test_groq_chat_overrides_provider(monkeypatch):
    _patch(monkeypatch, post=_Resp(OK))
    resp = await GroqProvider(api_key="k").chat({"model": "llama-3.3-70b-versatile"})
    assert resp.provider == "groq"
    assert resp.content == "hi"


async def test_together_chat_overrides_provider(monkeypatch):
    _patch(monkeypatch, post=_Resp(OK))
    resp = await TogetherProvider(api_key="k").chat({"model": "x"})
    assert resp.provider == "together"
    assert resp.content == "hi"


async def test_mistral_chat_overrides_provider(monkeypatch):
    _patch(monkeypatch, post=_Resp(OK))
    resp = await MistralProvider(api_key="k").chat({"model": "x"})
    assert resp.provider == "mistral"
    assert resp.content == "hi"


# --- error propagation through the subclass chat() ------------------------

@pytest.mark.parametrize(
    "cls",
    [GroqProvider, TogetherProvider, MistralProvider],
)
async def test_chat_http_error_propagates(monkeypatch, cls):
    _patch(monkeypatch, post=_Resp(raise_exc=_status_error(429)))
    with pytest.raises(ProviderError) as ei:
        await cls(api_key="k").chat({"model": "x"})
    assert ei.value.retryable is True
    assert ei.value.status_code == 429


@pytest.mark.parametrize(
    "cls",
    [GroqProvider, TogetherProvider, MistralProvider],
)
async def test_chat_connect_error_propagates(monkeypatch, cls):
    _patch(monkeypatch, post_exc=httpx.ConnectError("down"))
    with pytest.raises(ProviderError) as ei:
        await cls(api_key="k").chat({"model": "x"})
    assert ei.value.retryable is True
