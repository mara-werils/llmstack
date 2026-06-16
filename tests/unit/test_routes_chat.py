"""Tests for the /v1/chat/completions API route."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.circuit_breaker import CircuitBreakerError
from llmstack.gateway.routes import chat as chat_route


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, name: str):
        self.name = name


class _FakeProfile:
    def __init__(self, tier: str = "simple", score: float = 0.1):
        self.tier = tier
        self.score = score


class _FakeDecision:
    def __init__(self, model: str, tier: str = "simple", provider: str = "local"):
        self.model = model
        self.profile = _FakeProfile(tier=tier)
        self.provider = provider
        self.estimated_speedup = 2.0


class _FakeRouter:
    """Stand-in for ModelRouter used by ``_try_route``."""

    def __init__(self, models=("fast", "smart"), decision_model="fast"):
        self.models = [_FakeModel(m) for m in models]
        self._decision_model = decision_model
        self.overrides = []
        self.routed_messages = None

    def override(self, model):
        self.overrides.append(model)

    def route(self, messages):
        self.routed_messages = messages
        return _FakeDecision(self._decision_model)


def _install_router(monkeypatch, router):
    """Patch the lazily-imported ``get_router`` at its source module."""
    import llmstack.gateway.router._state as state

    monkeypatch.setattr(state, "get_router", lambda: router)


def _make_client():
    app = FastAPI()
    app.include_router(chat_route.router, prefix="/v1")
    return TestClient(app)


def _body(model="auto", stream=False, **extra):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hello there"}],
        "stream": stream,
    }
    payload.update(extra)
    return payload


@pytest.fixture(autouse=True)
def _no_router(monkeypatch):
    """By default the router is absent so ``_try_route`` is a no-op."""
    import llmstack.gateway.router._state as state

    monkeypatch.setattr(state, "get_router", lambda: None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_payload_returns_422(self, monkeypatch):
        # messages missing entirely -> ValidationError branch
        async def _proxy(*a, **k):  # pragma: no cover - should not be called
            raise AssertionError("proxy should not run on invalid input")

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json={"model": "x"})
        assert resp.status_code == 422
        assert resp.json()["error"]["type"] == "validation_error"

    def test_bad_role_returns_422(self, monkeypatch):
        monkeypatch.setattr(
            chat_route, "proxy_chat_completion", lambda *a, **k: None
        )
        client = _make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "nope", "content": "hi"}]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Non-streaming completion
# ---------------------------------------------------------------------------


class TestNonStreaming:
    def test_basic_completion(self, monkeypatch):
        captured = {}

        async def _proxy(payload, stream, provider_name):
            captured["payload"] = payload
            captured["stream"] = stream
            captured["provider"] = provider_name
            return {
                "id": "cmpl-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="gpt"))
        assert resp.status_code == 200
        assert resp.json()["id"] == "cmpl-1"
        assert resp.headers["X-Cache"] == "MISS"
        assert captured["stream"] is False

    def test_cache_hit_headers(self, monkeypatch):
        async def _proxy(payload, stream, provider_name):
            return {
                "id": "cmpl-2",
                "choices": [{"message": {"content": "cached"}, "finish_reason": "stop"}],
                "_cached": True,
                "_cache_age_s": 42,
                "_cached_at": 123,
            }

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="gpt"))
        assert resp.status_code == 200
        assert resp.headers["X-Cache"] == "HIT"
        assert resp.headers["X-Cache-Age"] == "42"
        # Internal cache markers are stripped before serialization; only the
        # public payload is returned, with cache status conveyed via headers.
        body = resp.json()
        assert body["id"] == "cmpl-2"
        assert "_cached" not in body
        assert "_cached_at" not in body
        assert "_cache_age_s" not in body

    def test_cost_header_emitted(self, monkeypatch):
        async def _proxy(payload, stream, provider_name):
            return {
                "id": "cmpl-3",
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "x_llmstack": {"cost_usd": 0.00123},
            }

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="gpt"))
        assert resp.status_code == 200
        assert resp.headers["X-Cost-USD"] == "0.001230"


# ---------------------------------------------------------------------------
# Routing branches (_try_route / _record_stats / _record_trace)
# ---------------------------------------------------------------------------


class TestRouting:
    def test_auto_routes_to_decision_model(self, monkeypatch):
        router = _FakeRouter(models=("fast", "smart"), decision_model="fast")
        _install_router(monkeypatch, router)

        captured = {}

        async def _proxy(payload, stream, provider_name):
            captured["model"] = payload["model"]
            captured["provider"] = provider_name
            return {
                "id": "r1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="auto"))
        assert resp.status_code == 200
        # Router replaced the model and exposed routing headers
        assert captured["model"] == "fast"
        assert resp.headers["X-Model-Router"] == "fast"
        assert resp.headers["X-Query-Tier"] == "simple"
        assert resp.headers["X-Provider"] == "local"
        assert router.routed_messages is not None

    def test_known_model_sets_override(self, monkeypatch):
        router = _FakeRouter(models=("fast", "smart"), decision_model="smart")
        _install_router(monkeypatch, router)

        async def _proxy(payload, stream, provider_name):
            return {"id": "r2",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="fast"))
        assert resp.status_code == 200
        # Override set then cleared (None)
        assert "fast" in router.overrides
        assert None in router.overrides

    def test_unknown_model_resolves_provider(self, monkeypatch):
        router = _FakeRouter(models=("fast", "smart"))
        _install_router(monkeypatch, router)
        # An unknown model triggers _resolve_provider_for_model
        monkeypatch.setattr(
            chat_route, "_resolve_provider_for_model", lambda m: "openai"
        )

        captured = {}

        async def _proxy(payload, stream, provider_name):
            captured["provider"] = provider_name
            captured["model"] = payload["model"]
            return {"id": "r3",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="gpt-4o-mini"))
        assert resp.status_code == 200
        # Unknown model is NOT mutated by the router, provider is resolved
        assert captured["model"] == "gpt-4o-mini"
        assert captured["provider"] == "openai"
        assert resp.headers["X-Provider"] == "openai"
        # No routed_model -> no X-Model-Router header
        assert "X-Model-Router" not in resp.headers

    def test_router_import_failure_is_silent(self, monkeypatch):
        # Force the lazy import to blow up -> _try_route returns untouched payload
        import builtins

        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "llmstack.gateway.router._state":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)

        async def _proxy(payload, stream, provider_name):
            return {"id": "r4",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="auto"))
        assert resp.status_code == 200
        assert "X-Model-Router" not in resp.headers


# ---------------------------------------------------------------------------
# _resolve_provider_for_model
# ---------------------------------------------------------------------------


class TestResolveProvider:
    def test_returns_none_when_no_registry(self, monkeypatch):
        import llmstack.gateway.providers.registry as reg

        monkeypatch.setattr(reg, "get_registry", lambda: None)
        assert chat_route._resolve_provider_for_model("foo") is None

    def test_returns_provider_name_for_known_model(self, monkeypatch):
        import llmstack.gateway.providers.registry as reg

        class _Prov:
            name = "anthropic"

        class _Reg:
            def get_provider_for_model(self, m):
                return _Prov()

            def _guess_provider(self, m):  # pragma: no cover - not reached
                return None

        monkeypatch.setattr(reg, "get_registry", lambda: _Reg())
        assert chat_route._resolve_provider_for_model("claude") == "anthropic"

    def test_falls_back_to_guess(self, monkeypatch):
        import llmstack.gateway.providers.registry as reg

        class _Prov:
            name = "openai"

        class _Reg:
            def get_provider_for_model(self, m):
                return None

            def _guess_provider(self, m):
                return _Prov()

        monkeypatch.setattr(reg, "get_registry", lambda: _Reg())
        assert chat_route._resolve_provider_for_model("gpt-x") == "openai"

    def test_guess_returns_none(self, monkeypatch):
        import llmstack.gateway.providers.registry as reg

        class _Reg:
            def get_provider_for_model(self, m):
                return None

            def _guess_provider(self, m):
                return None

        monkeypatch.setattr(reg, "get_registry", lambda: _Reg())
        assert chat_route._resolve_provider_for_model("???") is None

    def test_import_error_is_swallowed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "llmstack.gateway.providers.registry":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        assert chat_route._resolve_provider_for_model("foo") is None


# ---------------------------------------------------------------------------
# Streaming completion
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_returns_sse_chunks(self, monkeypatch):
        async def _gen():
            yield "data: chunk-1\n\n"
            yield "data: chunk-2\n\n"
            yield "data: [DONE]\n\n"

        async def _proxy(payload, stream, provider_name):
            assert stream is True
            return _gen()

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        with client.stream(
            "POST", "/v1/chat/completions", json=_body(model="gpt", stream=True)
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())
        assert "chunk-1" in body
        assert "chunk-2" in body
        assert "[DONE]" in body

    def test_stream_sets_routing_headers(self, monkeypatch):
        router = _FakeRouter(models=("fast", "smart"), decision_model="fast")
        _install_router(monkeypatch, router)

        async def _gen():
            yield "data: x\n\n"

        async def _proxy(payload, stream, provider_name):
            return _gen()

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        with client.stream(
            "POST", "/v1/chat/completions", json=_body(model="auto", stream=True)
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["X-Model-Router"] == "fast"
            assert resp.headers["X-Query-Tier"] == "simple"
            assert resp.headers["X-Provider"] == "local"
            assert resp.headers["Cache-Control"] == "no-cache"
            "".join(resp.iter_text())


# ---------------------------------------------------------------------------
# Error branches
# ---------------------------------------------------------------------------


class TestErrors:
    def test_circuit_breaker_open_returns_503(self, monkeypatch):
        async def _proxy(payload, stream, provider_name):
            raise CircuitBreakerError(retry_after=12.4)

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="gpt"))
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["type"] == "service_unavailable"
        assert body["error"]["retry_after"] == 12
        assert resp.headers["Retry-After"] == "12"

    def test_connect_error_returns_502(self, monkeypatch):
        async def _proxy(payload, stream, provider_name):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post("/v1/chat/completions", json=_body(model="gpt"))
        assert resp.status_code == 502
        assert resp.json()["error"]["type"] == "bad_gateway"

    def test_circuit_breaker_open_on_stream(self, monkeypatch):
        async def _proxy(payload, stream, provider_name):
            raise CircuitBreakerError(retry_after=5.0)

        monkeypatch.setattr(chat_route, "proxy_chat_completion", _proxy)
        client = _make_client()
        resp = client.post(
            "/v1/chat/completions", json=_body(model="gpt", stream=True)
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Observability helpers exercised directly
# ---------------------------------------------------------------------------


class TestObservabilityHelpers:
    def test_record_trace_adds_trace_and_records_quality(self, monkeypatch):
        import llmstack.observe._state as obs

        added = []
        recorded = []

        class _Store:
            def add(self, trace):
                added.append(trace)

        class _Score:
            def to_dict(self):
                return {"overall": 0.9}

        class _Scorer:
            def score(self, query, response):
                return _Score()

        class _Tracker:
            def record(self, quality, model, provider):
                recorded.append((quality, model, provider))

        monkeypatch.setattr(obs, "get_trace_store", lambda: _Store())
        monkeypatch.setattr(obs, "get_scorer", lambda: _Scorer())
        monkeypatch.setattr(obs, "get_tracker", lambda: _Tracker())

        result = {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        payload = {"messages": [{"role": "user", "content": "q?"}], "temperature": 0.5}
        chat_route._record_trace(
            payload, result, "fast", "local", "simple", 10.0, 0.0, cached=False
        )
        assert len(added) == 1
        assert recorded and recorded[0][1] == "fast"

    def test_record_trace_no_store_is_noop(self, monkeypatch):
        import llmstack.observe._state as obs

        monkeypatch.setattr(obs, "get_trace_store", lambda: None)
        monkeypatch.setattr(obs, "get_scorer", lambda: None)
        monkeypatch.setattr(obs, "get_tracker", lambda: None)
        # Should not raise even though there's nothing to record
        chat_route._record_trace({}, {}, "m", None, None, 1.0, 0.0)

    def test_record_trace_swallows_exceptions(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "llmstack.observe._state":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        # Import failure must be swallowed silently
        chat_route._record_trace({}, {}, "m", None, None, 1.0, 0.0)

    def test_record_stats_noop_when_model_none(self):
        # model is None -> early return, nothing imported
        chat_route._record_stats(None, "simple", 5.0)

    def test_record_stats_records_decision(self, monkeypatch):
        import llmstack.gateway.router._state as state

        recorded = []

        class _Stats:
            def record(self, decision, latency_ms, cost_usd):
                recorded.append((decision, latency_ms, cost_usd))

        monkeypatch.setattr(state, "get_stats", lambda: _Stats())
        chat_route._record_stats("fast", "simple", 12.0, provider="openai", cost_usd=0.01)
        assert recorded
        decision, latency, cost = recorded[0]
        assert decision.model == "fast"
        assert latency == 12.0
        assert cost == 0.01

    def test_record_stats_no_stats_is_noop(self, monkeypatch):
        import llmstack.gateway.router._state as state

        monkeypatch.setattr(state, "get_stats", lambda: None)
        chat_route._record_stats("fast", "simple", 1.0)

    def test_record_stats_swallows_exceptions(self, monkeypatch):
        import llmstack.gateway.router._state as state

        def _boom():
            raise RuntimeError("kaboom")

        monkeypatch.setattr(state, "get_stats", _boom)
        # Exception inside the body must be swallowed by the bare except.
        chat_route._record_stats("fast", "simple", 1.0)
