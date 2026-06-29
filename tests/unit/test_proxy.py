"""Tests for the gateway proxy layer."""

from __future__ import annotations

import httpx
import pytest

from llmstack.gateway import proxy
from llmstack.gateway.providers import registry as registry_mod

CHAT_RESULT = {"usage": {"prompt_tokens": 5, "completion_tokens": 2}, "choices": []}


class _FakeBreaker:
    def __init__(self):
        self.state = "closed"
        self.successes = 0
        self.failures = 0

    def check(self):
        pass

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class _FakeCache:
    def __init__(self, preset=None):
        self._store = {}
        self._preset = preset
        self.puts = 0

    async def get(self, model, messages, temperature):
        return self._preset

    async def put(self, model, messages, result, temperature):
        self.puts += 1


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
    def __init__(self, chunks, raise_exc=None, mid_exc=None):
        self._chunks = chunks
        self._raise_exc = raise_exc
        self._mid_exc = mid_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c
        if self._mid_exc:
            raise self._mid_exc


class _FakePool:
    def __init__(
        self,
        *,
        post=None,
        get=None,
        post_exc=None,
        stream_chunks=None,
        stream_exc=None,
        stream_mid_exc=None,
    ):
        self._post = post
        self._get = get
        self._post_exc = post_exc
        self._stream_chunks = stream_chunks or []
        self._stream_exc = stream_exc
        self._stream_mid_exc = stream_mid_exc

    async def post(self, url, json=None):
        if self._post_exc:
            raise self._post_exc
        return self._post

    async def get(self, url):
        return self._get

    def stream(self, method, url, json=None):
        if self._stream_exc and not self._stream_chunks:
            return _StreamCtx([], raise_exc=self._stream_exc)
        return _StreamCtx(self._stream_chunks, mid_exc=self._stream_mid_exc)


@pytest.fixture
def wiring(monkeypatch):
    breaker = _FakeBreaker()
    cache = _FakeCache()
    recorded = []
    monkeypatch.setattr(proxy, "get_inference_breaker", lambda: breaker)

    async def _get_cache():
        return cache

    monkeypatch.setattr(proxy, "get_cache", _get_cache)
    monkeypatch.setattr(proxy, "record_tokens", lambda **kw: recorded.append(kw))
    return breaker, cache, recorded


class TestPoolLifecycle:
    async def test_get_pool_is_singleton(self):
        await proxy.close_pool()
        p1 = proxy._get_pool()
        p2 = proxy._get_pool()
        assert p1 is p2
        await proxy.close_pool()
        assert proxy._pool is None


class TestChatCompletion:
    async def test_success_records_and_caches(self, wiring, monkeypatch):
        breaker, cache, recorded = wiring
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(post=_Resp(CHAT_RESULT)))
        result = await proxy.proxy_chat_completion({"model": "m", "messages": []})
        assert result == CHAT_RESULT
        assert breaker.successes == 1
        assert cache.puts == 1
        assert recorded == [{"input_tokens": 5, "output_tokens": 2}]

    async def test_cache_hit_short_circuits(self, monkeypatch):
        breaker = _FakeBreaker()
        cache = _FakeCache(preset={"cached": True})
        monkeypatch.setattr(proxy, "get_inference_breaker", lambda: breaker)

        async def _get_cache():
            return cache

        monkeypatch.setattr(proxy, "get_cache", _get_cache)
        called = []
        monkeypatch.setattr(proxy, "_get_pool", lambda: called.append(1))
        result = await proxy.proxy_chat_completion({"model": "m", "messages": []})
        assert result == {"cached": True}
        assert called == []  # never hit the backend

    async def test_5xx_trips_breaker_and_raises(self, wiring, monkeypatch):
        breaker, cache, _ = wiring
        err = httpx.HTTPStatusError("e", request=None, response=httpx.Response(503))
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(post=_Resp(raise_exc=err)))
        with pytest.raises(httpx.HTTPStatusError):
            await proxy.proxy_chat_completion({"model": "m", "messages": []})
        assert breaker.failures == 1

    async def test_4xx_does_not_trip_breaker(self, wiring, monkeypatch):
        breaker, _, _ = wiring
        err = httpx.HTTPStatusError("e", request=None, response=httpx.Response(404))
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(post=_Resp(raise_exc=err)))
        with pytest.raises(httpx.HTTPStatusError):
            await proxy.proxy_chat_completion({"model": "m", "messages": []})
        assert breaker.failures == 0

    async def test_connect_error_trips_breaker(self, wiring, monkeypatch):
        breaker, _, _ = wiring
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(post_exc=httpx.ConnectError("x")))
        with pytest.raises(httpx.ConnectError):
            await proxy.proxy_chat_completion({"model": "m", "messages": []})
        assert breaker.failures == 1

    async def test_stream_path_yields_chunks(self, wiring, monkeypatch):
        breaker, _, _ = wiring
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(stream_chunks=[b"x", b"y"]))
        gen = await proxy.proxy_chat_completion({"model": "m"}, stream=True)
        chunks = [c async for c in gen]
        assert chunks == [b"x", b"y"]
        assert breaker.successes == 1

    async def test_stream_connect_error_trips_breaker(self, wiring, monkeypatch):
        breaker, _, _ = wiring
        monkeypatch.setattr(
            proxy, "_get_pool", lambda: _FakePool(stream_exc=httpx.ReadTimeout("t"))
        )
        gen = await proxy.proxy_chat_completion({"model": "m"}, stream=True)
        with pytest.raises(httpx.ReadTimeout):
            async for _ in gen:
                pass
        assert breaker.failures == 1

    async def test_stream_failure_after_first_chunk_is_not_a_success(self, wiring, monkeypatch):
        breaker, _, _ = wiring
        monkeypatch.setattr(
            proxy,
            "_get_pool",
            lambda: _FakePool(stream_chunks=[b"x"], stream_mid_exc=httpx.ReadTimeout("t")),
        )
        gen = await proxy.proxy_chat_completion({"model": "m"}, stream=True)
        with pytest.raises(httpx.ReadTimeout):
            async for _ in gen:
                pass
        # A stream that dies mid-flight is a failure only -- not also a success.
        assert breaker.failures == 1
        assert breaker.successes == 0


class TestProviderRouting:
    async def test_provider_chat_with_fallback(self, wiring, monkeypatch):
        _, cache, recorded = wiring

        class _Reg:
            async def chat_with_fallback(self, payload):
                return CHAT_RESULT

        monkeypatch.setattr(registry_mod, "get_registry", lambda: _Reg())
        result = await proxy.proxy_chat_completion(
            {"model": "m", "messages": []}, provider_name="openai"
        )
        assert result == CHAT_RESULT
        assert cache.puts == 1
        assert recorded[0]["input_tokens"] == 5

    async def test_provider_registry_missing_raises(self, monkeypatch):
        monkeypatch.setattr(registry_mod, "get_registry", lambda: None)
        with pytest.raises(RuntimeError):
            await proxy.proxy_chat_completion({"model": "m"}, provider_name="openai")

    async def test_provider_stream(self, monkeypatch):
        class _Reg:
            def stream_with_fallback(self, payload):
                async def gen():
                    yield b"z"

                return gen()

        monkeypatch.setattr(registry_mod, "get_registry", lambda: _Reg())
        gen = await proxy.proxy_chat_completion(
            {"model": "m"}, stream=True, provider_name="anthropic"
        )
        assert [c async for c in gen] == [b"z"]

    async def test_provider_cache_hit(self, monkeypatch):
        cache = _FakeCache(preset={"hit": 1})

        async def _get_cache():
            return cache

        monkeypatch.setattr(proxy, "get_cache", _get_cache)
        monkeypatch.setattr(registry_mod, "get_registry", lambda: object())
        result = await proxy.proxy_chat_completion(
            {"model": "m", "messages": []}, provider_name="openai"
        )
        assert result == {"hit": 1}


class TestEmbeddingsAndModels:
    async def test_proxy_embeddings(self, monkeypatch):
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(post=_Resp({"data": [1]})))
        assert await proxy.proxy_embeddings({"input": "x"}) == {"data": [1]}

    async def test_proxy_models(self, monkeypatch):
        monkeypatch.setattr(proxy, "_get_pool", lambda: _FakePool(get=_Resp({"data": []})))
        assert await proxy.proxy_models() == {"data": []}
