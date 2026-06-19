"""Unit tests for llmstack.gateway.providers.registry.

Covers FallbackChain helpers, ProviderRegistry registration / lookup /
model-map / fallback-chain building, refresh_models, chat_with_fallback and
stream_with_fallback failover branches, prefix-based provider guessing, and
the module-level init_registry/get_registry singleton helpers.

All provider clients are mocked — no real network access.
"""

from __future__ import annotations

import pytest

from llmstack.gateway.providers.base import (
    Provider,
    ProviderError,
    ProviderModel,
    ProviderResponse,
)
from llmstack.gateway.providers.registry import (
    FallbackChain,
    ProviderRegistry,
    get_registry,
    init_registry,
)


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockProvider(Provider):
    """Test provider returning canned chat / stream / model responses."""

    name = "mock"

    def __init__(
        self,
        *,
        should_fail: bool = False,
        retryable: bool = True,
        error_cls: type[Exception] = ProviderError,
        list_raises: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._should_fail = should_fail
        self._retryable = retryable
        self._error_cls = error_cls
        self._list_raises = list_raises
        self.chat_calls = 0
        self.stream_calls = 0
        self._models = [
            ProviderModel(
                id="mock-small",
                provider="mock",
                cost_per_m_input=0.10,
                cost_per_m_output=0.30,
            ),
            ProviderModel(
                id="mock-large",
                provider="mock",
                cost_per_m_input=5.00,
                cost_per_m_output=15.00,
            ),
        ]

    def _raise(self, msg: str):
        if self._error_cls is ProviderError:
            raise ProviderError(msg, retryable=self._retryable)
        raise self._error_cls(msg)

    async def chat(self, payload: dict) -> ProviderResponse:
        self.chat_calls += 1
        if self._should_fail:
            self._raise("mock chat failure")
        model = payload.get("model", "mock-small")
        return ProviderResponse(
            content="hello from mock",
            model=model,
            provider=self.name,
            input_tokens=10,
            output_tokens=5,
            latency_ms=42.0,
            cost_usd=self.calculate_cost(model, 10, 5),
        )

    async def chat_stream(self, payload: dict):
        self.stream_calls += 1
        if self._should_fail:
            self._raise("mock stream failure")
        yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def list_models(self) -> list[ProviderModel]:
        if self._list_raises:
            raise RuntimeError("boom listing models")
        return self._models


class MockProvider2(MockProvider):
    name = "mock2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._models = [
            ProviderModel(
                id="mock2-medium",
                provider="mock2",
                cost_per_m_input=1.00,
                cost_per_m_output=3.00,
            ),
        ]


def _make_provider(name: str, **kwargs) -> MockProvider:
    p = MockProvider(**kwargs)
    p.name = name
    return p


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_len(self):
        chain = FallbackChain(steps=[("openai", "gpt-4o"), ("anthropic", "claude-3")])
        assert len(chain) == 2

    def test_empty_len(self):
        assert len(FallbackChain(steps=[])) == 0

    def test_is_empty_true(self):
        assert FallbackChain(steps=[]).is_empty is True

    def test_is_empty_false(self):
        assert FallbackChain(steps=[("openai", "gpt-4o")]).is_empty is False

    def test_primary_present(self):
        chain = FallbackChain(steps=[("openai", "gpt-4o"), ("anthropic", "claude-3")])
        assert chain.primary == ("openai", "gpt-4o")

    def test_primary_none_when_empty(self):
        assert FallbackChain(steps=[]).primary is None


# ---------------------------------------------------------------------------
# Registration / counts / lookup
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_get_provider(self):
        r = ProviderRegistry()
        p = MockProvider()
        r.register(p)
        assert r.get_provider("mock") is p

    def test_get_unknown_provider_returns_none(self):
        assert ProviderRegistry().get_provider("nope") is None

    def test_provider_count(self):
        r = ProviderRegistry()
        assert r.provider_count == 0
        r.register(MockProvider())
        r.register(MockProvider2())
        assert r.provider_count == 2

    def test_model_count_reflects_all_models(self):
        r = ProviderRegistry()
        assert r.model_count == 0
        r._all_models = [
            ProviderModel(id="a", provider="mock"),
            ProviderModel(id="b", provider="mock"),
        ]
        assert r.model_count == 2

    def test_has_provider_true_and_false(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        assert r.has_provider("mock") is True
        assert r.has_provider("missing") is False

    def test_register_model_and_lookup(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register_model("gpt-4o", "mock")
        p = r.get_provider_for_model("gpt-4o")
        assert p is not None and p.name == "mock"

    def test_get_provider_for_unknown_model(self):
        assert ProviderRegistry().get_provider_for_model("unknown") is None

    def test_get_provider_for_model_mapped_to_missing_provider(self):
        # model is mapped, but provider instance not registered -> None
        r = ProviderRegistry()
        r.register_model("ghost", "ghost-provider")
        assert r.get_provider_for_model("ghost") is None

    def test_all_providers_returns_copy(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        providers = r.all_providers()
        assert "mock" in providers
        providers["injected"] = object()
        assert "injected" not in r.all_providers()

    def test_all_models_returns_copy(self):
        r = ProviderRegistry()
        r._all_models = [ProviderModel(id="x", provider="mock")]
        models = r.all_models()
        assert [m.id for m in models] == ["x"]
        models.append(ProviderModel(id="y", provider="mock"))
        assert len(r.all_models()) == 1


# ---------------------------------------------------------------------------
# refresh_models
# ---------------------------------------------------------------------------


class TestRefreshModels:
    async def test_refresh_aggregates_and_builds_map(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        models = await r.refresh_models()
        ids = {m.id for m in models}
        assert ids == {"mock-small", "mock-large", "mock2-medium"}
        assert r.model_count == 3
        p = r.get_provider_for_model("mock2-medium")
        assert p is not None and p.name == "mock2"

    async def test_refresh_sets_provider_on_each_model(self):
        r = ProviderRegistry()
        r.register(MockProvider2())
        models = await r.refresh_models()
        assert all(m.provider == "mock2" for m in models)

    async def test_refresh_is_idempotent_resets_all_models(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        await r.refresh_models()
        first = r.model_count
        # second call should not double-count
        await r.refresh_models()
        assert r.model_count == first

    async def test_refresh_swallows_provider_errors(self, caplog):
        r = ProviderRegistry()
        r.register(_make_provider("good"))
        r.register(_make_provider("bad", list_raises=True))
        models = await r.refresh_models()
        # only the good provider's models survive; the failing one is skipped
        assert {m.id for m in models} == {"mock-small", "mock-large"}
        assert "Failed to list models" in caplog.text


# ---------------------------------------------------------------------------
# get_fallback_chain
# ---------------------------------------------------------------------------


class TestFallbackChainBuilding:
    @pytest.fixture
    def registry(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        r.register_model("mock-small", "mock")
        r.register_model("mock2-medium", "mock2")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]
        return r

    def test_chain_with_fallback(self, registry):
        chain = registry.get_fallback_chain("mock-small")
        assert chain.steps[0] == ("mock", "mock-small")
        assert chain.steps[1] == ("mock2", "mock2-medium")
        assert len(chain) == 2

    def test_chain_without_fallback(self, registry):
        chain = registry.get_fallback_chain("mock2-medium")
        assert chain.steps == [("mock2", "mock2-medium")]

    def test_unknown_model_empty_chain(self, registry):
        chain = registry.get_fallback_chain("nope")
        assert chain.is_empty

    def test_fallback_provider_not_registered_is_skipped(self):
        # "ghost" is configured as a fallback but never registered -> continue (line 105)
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["ghost"])
        r._all_models = [ProviderModel(id="mock-small", provider="mock")]
        chain = r.get_fallback_chain("mock-small")
        assert chain.steps == [("mock", "mock-small")]

    def test_fallback_provider_registered_but_no_models(self):
        # fallback provider exists but contributes no models -> no extra step
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [ProviderModel(id="mock-small", provider="mock")]
        chain = r.get_fallback_chain("mock-small")
        assert chain.steps == [("mock", "mock-small")]


# ---------------------------------------------------------------------------
# chat_with_fallback
# ---------------------------------------------------------------------------


def _chain_registry(primary_fail=False, retryable=True, error_cls=ProviderError):
    r = ProviderRegistry()
    primary = _make_provider(
        "mock",
        should_fail=primary_fail,
        retryable=retryable,
        error_cls=error_cls,
    )
    fallback = MockProvider2()
    r.register(primary)
    r.register(fallback)
    r.register_model("mock-small", "mock")
    r.register_model("mock2-medium", "mock2")
    r.set_fallbacks("mock", ["mock2"])
    r._all_models = [
        ProviderModel(id="mock-small", provider="mock"),
        ProviderModel(id="mock2-medium", provider="mock2"),
    ]
    return r, primary, fallback


class TestChatWithFallback:
    async def test_primary_success_no_fallback_flag(self):
        r, _, _ = _chain_registry()
        result = await r.chat_with_fallback(
            {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["x_llmstack"]["provider"] == "mock"
        assert result["x_llmstack"]["model"] == "mock-small"
        assert result["x_llmstack"]["fallback"] is False
        assert result["x_llmstack"]["cost_usd"] >= 0.0
        assert "latency_ms" in result["x_llmstack"]

    async def test_retryable_failure_triggers_fallback(self):
        r, primary, fallback = _chain_registry(primary_fail=True, retryable=True)
        result = await r.chat_with_fallback(
            {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["x_llmstack"]["provider"] == "mock2"
        assert result["x_llmstack"]["fallback"] is True
        assert primary.chat_calls == 1
        assert fallback.chat_calls == 1

    async def test_non_retryable_error_reraises_immediately(self):
        r, _, fallback = _chain_registry(primary_fail=True, retryable=False)
        with pytest.raises(ProviderError):
            await r.chat_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )
        assert fallback.chat_calls == 0  # fallback never reached

    async def test_generic_exception_triggers_fallback(self):
        # non-ProviderError exception path (lines 182-189)
        r, primary, fallback = _chain_registry(primary_fail=True, error_cls=RuntimeError)
        result = await r.chat_with_fallback(
            {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["x_llmstack"]["provider"] == "mock2"
        assert primary.chat_calls == 1

    async def test_all_providers_fail_raises(self):
        r = ProviderRegistry()
        f1 = _make_provider("mock", should_fail=True, retryable=True)
        f2 = _make_provider("mock2", should_fail=True, retryable=True)
        r.register(f1)
        r.register(f2)
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]
        with pytest.raises(ProviderError, match="All providers failed"):
            await r.chat_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )

    async def test_no_provider_but_prefix_guess_succeeds(self):
        # empty chain -> _guess_provider routes by "gpt-" prefix (lines 147-149)
        r = ProviderRegistry()
        r.register(_make_provider("openai"))
        result = await r.chat_with_fallback(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["x_llmstack"]["provider"] == "openai"
        assert result["x_llmstack"]["model"] == "gpt-4o"

    async def test_no_provider_and_no_guess_raises(self):
        # empty chain and no prefix match (lines 150-151)
        r = ProviderRegistry()
        with pytest.raises(ProviderError, match="No provider found"):
            await r.chat_with_fallback({"model": "totally-unknown-model"})

    async def test_chain_step_provider_missing_is_skipped(self):
        # model_map points at "mock"+fallback that was registered, but the
        # primary provider instance is removed -> step skipped (line 157).
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]
        # drop the primary provider instance after the chain is configured
        del r._providers["mock"]
        result = await r.chat_with_fallback(
            {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
        )
        # first step (mock) skipped, fallback (mock2) handles it
        assert result["x_llmstack"]["provider"] == "mock2"

    async def test_missing_model_key_defaults_to_empty(self):
        r = ProviderRegistry()
        with pytest.raises(ProviderError, match="No provider found for model ''"):
            await r.chat_with_fallback({"messages": []})


# ---------------------------------------------------------------------------
# stream_with_fallback
# ---------------------------------------------------------------------------


async def _collect(agen):
    return [chunk async for chunk in agen]


class TestStreamWithFallback:
    async def test_stream_primary_success(self):
        r, _, _ = _chain_registry()
        chunks = await _collect(
            r.stream_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )
        )
        assert chunks[-1] == b"data: [DONE]\n\n"

    async def test_stream_retryable_failure_triggers_fallback(self):
        r, primary, fallback = _chain_registry(primary_fail=True, retryable=True)
        chunks = await _collect(
            r.stream_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )
        )
        assert b"data: [DONE]\n\n" in chunks
        assert primary.stream_calls == 1
        assert fallback.stream_calls == 1

    async def test_stream_non_retryable_reraises(self):
        r, _, fallback = _chain_registry(primary_fail=True, retryable=False)
        with pytest.raises(ProviderError):
            await _collect(
                r.stream_with_fallback(
                    {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
                )
            )
        assert fallback.stream_calls == 0

    async def test_stream_generic_exception_triggers_fallback(self):
        r, primary, fallback = _chain_registry(primary_fail=True, error_cls=RuntimeError)
        chunks = await _collect(
            r.stream_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )
        )
        assert b"data: [DONE]\n\n" in chunks
        assert fallback.stream_calls == 1

    async def test_stream_all_fail_raises(self):
        r = ProviderRegistry()
        f1 = _make_provider("mock", should_fail=True, retryable=True)
        f2 = _make_provider("mock2", should_fail=True, retryable=True)
        r.register(f1)
        r.register(f2)
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]
        with pytest.raises(ProviderError, match="All providers failed streaming"):
            await _collect(
                r.stream_with_fallback(
                    {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
                )
            )

    async def test_stream_no_provider_but_guess_succeeds(self):
        r = ProviderRegistry()
        r.register(_make_provider("anthropic"))
        chunks = await _collect(
            r.stream_with_fallback(
                {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]}
            )
        )
        assert b"data: [DONE]\n\n" in chunks

    async def test_stream_no_provider_no_guess_raises(self):
        r = ProviderRegistry()
        with pytest.raises(ProviderError, match="No provider found"):
            await _collect(r.stream_with_fallback({"model": "weird-model"}))

    async def test_stream_chain_step_provider_missing_skipped(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]
        del r._providers["mock"]
        chunks = await _collect(
            r.stream_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )
        )
        assert b"data: [DONE]\n\n" in chunks


# ---------------------------------------------------------------------------
# _guess_provider
# ---------------------------------------------------------------------------


class TestGuessProvider:
    @pytest.mark.parametrize(
        "model_id,provider_name",
        [
            ("gpt-4o", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-test", "openai"),
            ("claude-sonnet-4-20250514", "anthropic"),
            ("gemini-2.5-flash", "google"),
            ("mistral-large", "mistral"),
            ("codestral-latest", "mistral"),
            ("pixtral-12b", "mistral"),
            ("llama3.3-70b", "groq"),
            ("mixtral-8x7b", "groq"),
        ],
    )
    def test_prefix_matches(self, model_id, provider_name):
        r = ProviderRegistry()
        r.register(_make_provider(provider_name))
        p = r._guess_provider(model_id)
        assert p is not None and p.name == provider_name

    def test_prefix_matches_but_provider_not_registered(self):
        # prefix resolves to "openai" but it's not registered -> None
        assert ProviderRegistry()._guess_provider("gpt-4o") is None

    def test_unknown_prefix_returns_none(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        assert r._guess_provider("some-random-model") is None


# ---------------------------------------------------------------------------
# module-level singleton helpers
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_init_and_get_registry(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        init_registry(r)
        assert get_registry() is r

    def test_init_registry_overwrites(self):
        first = ProviderRegistry()
        second = ProviderRegistry()
        init_registry(first)
        assert get_registry() is first
        init_registry(second)
        assert get_registry() is second
