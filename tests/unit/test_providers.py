"""Comprehensive tests for the Universal LLM Gateway provider system.

Covers provider base class, registry, fallback chains, cost routing,
format translation (Anthropic, Google), and stats cost tracking.
"""

from __future__ import annotations

import pytest

from llmstack.gateway.providers.base import Provider, ProviderError, ProviderModel, ProviderResponse
from llmstack.gateway.providers.registry import ProviderRegistry, FallbackChain
from llmstack.gateway.providers.anthropic_provider import (
    _openai_to_anthropic,
    _anthropic_to_openai,
)
from llmstack.gateway.router.router import ModelRouter, ModelTier, RoutingDecision
from llmstack.gateway.router.classifier import QueryProfile
from llmstack.gateway.router.stats import RouterStats


# ---------------------------------------------------------------------------
# Mock provider for testing
# ---------------------------------------------------------------------------

class MockProvider(Provider):
    """A test provider that returns canned responses."""

    name = "mock"

    def __init__(self, should_fail: bool = False, retryable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._should_fail = should_fail
        self._retryable = retryable
        self._call_count = 0
        self._models = [
            ProviderModel(id="mock-small", provider="mock",
                          cost_per_m_input=0.10, cost_per_m_output=0.30),
            ProviderModel(id="mock-large", provider="mock",
                          cost_per_m_input=5.00, cost_per_m_output=15.00),
        ]

    async def chat(self, payload: dict) -> ProviderResponse:
        self._call_count += 1
        if self._should_fail:
            raise ProviderError("Mock failure", retryable=self._retryable)
        model = payload.get("model", "mock-small")
        return ProviderResponse(
            content="Hello from mock!",
            model=model,
            provider="mock",
            input_tokens=10,
            output_tokens=5,
            latency_ms=50.0,
            cost_usd=self.calculate_cost(model, 10, 5),
        )

    async def chat_stream(self, payload: dict):
        if self._should_fail:
            raise ProviderError("Mock stream failure", retryable=self._retryable)
        yield b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n"
        yield b"data: [DONE]\n\n"

    async def list_models(self) -> list[ProviderModel]:
        return self._models


class MockProvider2(MockProvider):
    name = "mock2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._models = [
            ProviderModel(id="mock2-medium", provider="mock2",
                          cost_per_m_input=1.00, cost_per_m_output=3.00),
        ]


# ===================================================================
# Provider base class tests
# ===================================================================

class TestProviderBase:
    def test_calculate_cost(self):
        p = MockProvider()
        # mock-small: $0.10/M input, $0.30/M output
        cost = p.calculate_cost("mock-small", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.40)

    def test_calculate_cost_small_request(self):
        p = MockProvider()
        cost = p.calculate_cost("mock-small", 100, 50)
        # 100 * 0.10 / 1M + 50 * 0.30 / 1M
        assert cost == pytest.approx(0.000025)

    def test_calculate_cost_unknown_model(self):
        p = MockProvider()
        cost = p.calculate_cost("nonexistent", 1000, 500)
        assert cost == 0.0

    def test_get_model_cost(self):
        p = MockProvider()
        ci, co = p.get_model_cost("mock-large")
        assert ci == 5.00
        assert co == 15.00

    def test_get_model_cost_unknown(self):
        p = MockProvider()
        ci, co = p.get_model_cost("unknown")
        assert ci == 0.0
        assert co == 0.0


class TestProviderResponse:
    def test_to_openai_dict_with_raw(self):
        resp = ProviderResponse(
            content="Hello",
            model="gpt-4o",
            provider="openai",
            raw={"id": "123", "choices": [{"message": {"content": "Hello"}}]},
        )
        result = resp.to_openai_dict()
        assert result["id"] == "123"

    def test_to_openai_dict_without_raw(self):
        resp = ProviderResponse(
            content="Hello",
            model="gpt-4o",
            provider="openai",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )
        result = resp.to_openai_dict()
        assert result["model"] == "gpt-4o"
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["x_llmstack"]["provider"] == "openai"
        assert result["x_llmstack"]["cost_usd"] == 0.001


class TestProviderError:
    def test_retryable_error(self):
        err = ProviderError("test", status_code=429, retryable=True)
        assert err.retryable is True
        assert err.status_code == 429

    def test_non_retryable_error(self):
        err = ProviderError("bad request", status_code=400, retryable=False)
        assert err.retryable is False


# ===================================================================
# Registry tests
# ===================================================================

class TestProviderRegistry:
    @pytest.fixture
    def registry(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        return r

    def test_register_and_get(self, registry):
        p = registry.get_provider("mock")
        assert p is not None
        assert p.name == "mock"

    def test_get_unknown_provider(self, registry):
        assert registry.get_provider("nonexistent") is None

    def test_register_model_mapping(self, registry):
        registry.register_model("gpt-4o", "mock")
        p = registry.get_provider_for_model("gpt-4o")
        assert p is not None
        assert p.name == "mock"

    def test_model_mapping_unknown(self, registry):
        assert registry.get_provider_for_model("unknown-model") is None

    @pytest.mark.asyncio
    async def test_refresh_models(self, registry):
        models = await registry.refresh_models()
        assert len(models) == 3  # 2 from mock + 1 from mock2
        ids = {m.id for m in models}
        assert "mock-small" in ids
        assert "mock-large" in ids
        assert "mock2-medium" in ids

    @pytest.mark.asyncio
    async def test_refresh_builds_model_map(self, registry):
        await registry.refresh_models()
        p = registry.get_provider_for_model("mock-small")
        assert p is not None
        assert p.name == "mock"

    def test_all_providers(self, registry):
        providers = registry.all_providers()
        assert "mock" in providers
        assert "mock2" in providers


# ===================================================================
# Fallback chain tests
# ===================================================================

class TestFallbackChain:
    def test_fallback_chain_length(self):
        chain = FallbackChain(steps=[("openai", "gpt-4o"), ("anthropic", "claude-3")])
        assert len(chain) == 2

    def test_empty_chain(self):
        chain = FallbackChain(steps=[])
        assert len(chain) == 0


class TestRegistryFallback:
    @pytest.fixture
    def registry_with_fallbacks(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        r.register(MockProvider2())
        r.register_model("mock-small", "mock")
        r.register_model("mock2-medium", "mock2")
        r.set_fallbacks("mock", ["mock2"])
        # Need to populate _all_models for fallback model lookup
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]
        return r

    def test_fallback_chain_built(self, registry_with_fallbacks):
        chain = registry_with_fallbacks.get_fallback_chain("mock-small")
        assert len(chain) == 2
        assert chain.steps[0] == ("mock", "mock-small")
        assert chain.steps[1] == ("mock2", "mock2-medium")

    def test_no_fallback_configured(self, registry_with_fallbacks):
        chain = registry_with_fallbacks.get_fallback_chain("mock2-medium")
        assert len(chain) == 1
        assert chain.steps[0] == ("mock2", "mock2-medium")

    def test_unknown_model_empty_chain(self, registry_with_fallbacks):
        chain = registry_with_fallbacks.get_fallback_chain("nonexistent")
        assert len(chain) == 0

    @pytest.mark.asyncio
    async def test_chat_with_fallback_success(self, registry_with_fallbacks):
        result = await registry_with_fallbacks.chat_with_fallback(
            {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["x_llmstack"]["provider"] == "mock"
        assert result["x_llmstack"]["fallback"] is False

    @pytest.mark.asyncio
    async def test_chat_with_fallback_triggers(self):
        """When primary fails, fallback provider should handle the request."""
        r = ProviderRegistry()
        failing = MockProvider(should_fail=True, retryable=True)
        failing.name = "mock"
        working = MockProvider2()

        r.register(failing)
        r.register(working)
        r.register_model("mock-small", "mock")
        r.register_model("mock2-medium", "mock2")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]

        result = await r.chat_with_fallback(
            {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["x_llmstack"]["provider"] == "mock2"
        assert result["x_llmstack"]["fallback"] is True

    @pytest.mark.asyncio
    async def test_chat_all_fail_raises(self):
        r = ProviderRegistry()
        failing1 = MockProvider(should_fail=True, retryable=True)
        failing1.name = "mock"
        failing2 = MockProvider(should_fail=True, retryable=True)
        failing2.name = "mock2"

        r.register(failing1)
        r.register(failing2)
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

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_fallback(self):
        """Non-retryable errors should not trigger fallback."""
        r = ProviderRegistry()
        failing = MockProvider(should_fail=True, retryable=False)
        failing.name = "mock"
        working = MockProvider2()

        r.register(failing)
        r.register(working)
        r.register_model("mock-small", "mock")
        r.set_fallbacks("mock", ["mock2"])
        r._all_models = [
            ProviderModel(id="mock-small", provider="mock"),
            ProviderModel(id="mock2-medium", provider="mock2"),
        ]

        with pytest.raises(ProviderError):
            await r.chat_with_fallback(
                {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
            )


# ===================================================================
# Provider guess tests
# ===================================================================

class TestProviderGuess:
    def test_guess_openai(self):
        r = ProviderRegistry()
        r.register(MockProvider())
        mock_openai = MockProvider()
        mock_openai.name = "openai"
        r.register(mock_openai)

        p = r._guess_provider("gpt-4o")
        assert p is not None
        assert p.name == "openai"

    def test_guess_anthropic(self):
        r = ProviderRegistry()
        mock = MockProvider()
        mock.name = "anthropic"
        r.register(mock)

        p = r._guess_provider("claude-sonnet-4-20250514")
        assert p is not None
        assert p.name == "anthropic"

    def test_guess_google(self):
        r = ProviderRegistry()
        mock = MockProvider()
        mock.name = "google"
        r.register(mock)

        p = r._guess_provider("gemini-2.5-flash")
        assert p is not None
        assert p.name == "google"

    def test_guess_unknown(self):
        r = ProviderRegistry()
        p = r._guess_provider("some-random-model")
        assert p is None


# ===================================================================
# Anthropic format translation tests
# ===================================================================

class TestAnthropicTranslation:
    def test_basic_translation(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        result = _openai_to_anthropic(payload)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 4096

    def test_system_message_extracted(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = _openai_to_anthropic(payload)
        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_multiple_system_messages(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "Rule 1"},
                {"role": "system", "content": "Rule 2"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = _openai_to_anthropic(payload)
        assert "Rule 1" in result["system"]
        assert "Rule 2" in result["system"]

    def test_temperature_preserved(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
        }
        result = _openai_to_anthropic(payload)
        assert result["temperature"] == 0.5

    def test_stop_sequences(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END", "STOP"],
        }
        result = _openai_to_anthropic(payload)
        assert result["stop_sequences"] == ["END", "STOP"]

    def test_stop_string(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }
        result = _openai_to_anthropic(payload)
        assert result["stop_sequences"] == ["END"]

    def test_consecutive_roles_merged(self):
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "World"},
            ],
        }
        result = _openai_to_anthropic(payload)
        assert len(result["messages"]) == 1
        assert "Hello" in result["messages"][0]["content"]
        assert "World" in result["messages"][0]["content"]

    def test_response_translation(self):
        anthropic_resp = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello there!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _anthropic_to_openai(anthropic_resp, "claude-sonnet-4-20250514", 100.0)
        assert result["choices"][0]["message"]["content"] == "Hello there!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5


# ===================================================================
# Cost-aware routing tests
# ===================================================================

class TestCostAwareRouting:
    @pytest.fixture
    def models_with_costs(self):
        return [
            ModelTier(name="gpt-4o-mini", tier="simple", speed_score=3.0,
                      quality_score=0.7, provider="openai",
                      cost_per_m_input=0.15, cost_per_m_output=0.60),
            ModelTier(name="llama3.2:1b", tier="simple", speed_score=2.5,
                      quality_score=0.6, provider="local",
                      cost_per_m_input=0.0, cost_per_m_output=0.0),
            ModelTier(name="gpt-4o", tier="medium", speed_score=1.5,
                      quality_score=0.9, provider="openai",
                      cost_per_m_input=2.50, cost_per_m_output=10.00),
            ModelTier(name="claude-sonnet-4-20250514", tier="medium", speed_score=1.2,
                      quality_score=0.95, provider="anthropic",
                      cost_per_m_input=3.00, cost_per_m_output=15.00),
            ModelTier(name="claude-opus-4-20250514", tier="complex", speed_score=0.5,
                      quality_score=1.0, provider="anthropic",
                      cost_per_m_input=15.00, cost_per_m_output=75.00),
        ]

    def test_cost_strategy_picks_free_local(self, models_with_costs):
        """Cost strategy should prefer free local models over paid cloud models."""
        router = ModelRouter(models=models_with_costs, strategy="cost")
        d = router.route([{"role": "user", "content": "hello"}])
        # Local model is free ($0.0) so it should be picked for simple queries
        assert d.model == "llama3.2:1b"
        assert d.provider == "local"

    def test_cost_strategy_medium_prefers_cheaper(self, models_with_costs):
        """For medium queries, cost strategy picks the cheaper cloud option."""
        router = ModelRouter(models=models_with_costs, strategy="cost")
        d = router.route([{"role": "user", "content":
            "Explain how neural networks learn through backpropagation and compare different optimizers"
        }])
        if d.profile.tier == "medium":
            # gpt-4o ($12.50/M total) is cheaper than claude-sonnet ($18/M total)
            assert d.model == "gpt-4o"
            assert d.provider == "openai"

    def test_quality_strategy_picks_best(self, models_with_costs):
        """Quality strategy ignores cost and picks highest quality."""
        router = ModelRouter(models=models_with_costs, strategy="quality")
        d = router.route([{"role": "user", "content": "hello"}])
        # Highest quality_score overall (with tier >= simple) is claude-opus (1.0)
        assert d.model == "claude-opus-4-20250514"
        assert d.provider == "anthropic"

    def test_routing_decision_has_provider(self, models_with_costs):
        """RoutingDecision should include provider name."""
        router = ModelRouter(models=models_with_costs, strategy="cost")
        d = router.route([{"role": "user", "content": "hello"}])
        assert d.provider in ("local", "openai", "anthropic")

    def test_routing_decision_has_cost(self, models_with_costs):
        """RoutingDecision should include estimated cost."""
        router = ModelRouter(models=models_with_costs, strategy="cost")
        d = router.route([{"role": "user", "content": "hello"}])
        assert hasattr(d, "estimated_cost_per_1k")


# ===================================================================
# Stats cost tracking tests
# ===================================================================

class TestStatsCostTracking:
    @pytest.fixture
    def stats(self):
        s = RouterStats()
        s.set_largest_model("claude-opus-4-20250514")
        return s

    def test_cost_tracking(self, stats):
        profile = QueryProfile(score=0.2, tier="simple", factors={})
        d = RoutingDecision(model="gpt-4o-mini", profile=profile, provider="openai")
        stats.record(d, latency_ms=50.0, cost_usd=0.001)

        s = stats.summary()
        assert s["total_cost_usd"] == pytest.approx(0.001)
        assert "openai" in s["cost_by_provider_usd"]
        assert s["cost_by_provider_usd"]["openai"] == pytest.approx(0.001)

    def test_multi_provider_cost_tracking(self, stats):
        # OpenAI request
        p1 = QueryProfile(score=0.2, tier="simple", factors={})
        d1 = RoutingDecision(model="gpt-4o-mini", profile=p1, provider="openai")
        stats.record(d1, latency_ms=50.0, cost_usd=0.001)

        # Anthropic request
        p2 = QueryProfile(score=0.8, tier="complex", factors={})
        d2 = RoutingDecision(model="claude-opus-4-20250514", profile=p2, provider="anthropic")
        stats.record(d2, latency_ms=200.0, cost_usd=0.05)

        # Local request (free)
        p3 = QueryProfile(score=0.1, tier="simple", factors={})
        d3 = RoutingDecision(model="llama3.2:1b", profile=p3, provider="local")
        stats.record(d3, latency_ms=30.0, cost_usd=0.0)

        s = stats.summary()
        assert s["total_cost_usd"] == pytest.approx(0.051)
        assert s["cost_by_provider_usd"]["openai"] == pytest.approx(0.001)
        assert s["cost_by_provider_usd"]["anthropic"] == pytest.approx(0.05)
        assert s["cost_by_provider_usd"]["local"] == pytest.approx(0.0)

    def test_provider_distribution(self, stats):
        for _ in range(3):
            p = QueryProfile(score=0.1, tier="simple", factors={})
            d = RoutingDecision(model="gpt-4o-mini", profile=p, provider="openai")
            stats.record(d, latency_ms=50.0)

        p = QueryProfile(score=0.1, tier="simple", factors={})
        d = RoutingDecision(model="llama3.2:1b", profile=p, provider="local")
        stats.record(d, latency_ms=30.0)

        s = stats.summary()
        assert s["provider_distribution"]["openai"]["count"] == 3
        assert s["provider_distribution"]["local"]["count"] == 1
        assert s["provider_distribution"]["openai"]["pct"] == pytest.approx(75.0)

    def test_recent_decisions_include_provider(self, stats):
        p = QueryProfile(score=0.5, tier="medium", factors={})
        d = RoutingDecision(model="gpt-4o", profile=p, provider="openai")
        stats.record(d, latency_ms=100.0, cost_usd=0.01)

        s = stats.summary()
        recent = s["recent_decisions"]
        assert len(recent) == 1
        assert recent[0]["provider"] == "openai"
        assert recent[0]["cost_usd"] == pytest.approx(0.01)


# ===================================================================
# OpenAI-compatible provider tests
# ===================================================================

class TestOpenAICompatProviders:
    def test_groq_has_models(self):
        from llmstack.gateway.providers.openai_compat import GroqProvider
        p = GroqProvider(api_key="test")
        assert p.name == "groq"
        assert len(p._models) > 0
        assert any("llama" in m.id for m in p._models)

    def test_together_has_models(self):
        from llmstack.gateway.providers.openai_compat import TogetherProvider
        p = TogetherProvider(api_key="test")
        assert p.name == "together"
        assert len(p._models) > 0

    def test_mistral_has_models(self):
        from llmstack.gateway.providers.openai_compat import MistralProvider
        p = MistralProvider(api_key="test")
        assert p.name == "mistral"
        assert len(p._models) > 0
        assert any("mistral" in m.id for m in p._models)

    def test_openai_has_pricing(self):
        from llmstack.gateway.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="test")
        assert p.name == "openai"
        ci, co = p.get_model_cost("gpt-4o")
        assert ci > 0
        assert co > 0

    def test_anthropic_has_pricing(self):
        from llmstack.gateway.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key="test")
        ci, co = p.get_model_cost("claude-sonnet-4-20250514")
        assert ci == 3.00
        assert co == 15.00

    def test_google_has_pricing(self):
        from llmstack.gateway.providers.google_provider import GoogleProvider
        p = GoogleProvider(api_key="test")
        ci, co = p.get_model_cost("gemini-2.5-flash")
        assert ci == 0.15
        assert co == 0.60


# ===================================================================
# Integration-style: router + providers combined
# ===================================================================

class TestRouterWithProviders:
    def test_multi_provider_routing(self):
        """Router should work across multiple providers."""
        models = [
            ModelTier(name="llama3.2:1b", tier="simple", provider="local",
                      speed_score=3.0, quality_score=0.5,
                      cost_per_m_input=0.0, cost_per_m_output=0.0),
            ModelTier(name="gpt-4o-mini", tier="simple", provider="openai",
                      speed_score=5.0, quality_score=0.7,
                      cost_per_m_input=0.15, cost_per_m_output=0.60),
            ModelTier(name="claude-sonnet-4-20250514", tier="medium", provider="anthropic",
                      speed_score=1.2, quality_score=0.95,
                      cost_per_m_input=3.00, cost_per_m_output=15.00),
        ]

        # Cost strategy should pick free local for simple
        router = ModelRouter(models=models, strategy="cost")
        d = router.route([{"role": "user", "content": "hi"}])
        assert d.model == "llama3.2:1b"
        assert d.provider == "local"

        # Latency strategy picks fastest for simple
        router = ModelRouter(models=models, strategy="latency")
        d = router.route([{"role": "user", "content": "hi"}])
        assert d.model == "gpt-4o-mini"
        assert d.provider == "openai"

    def test_model_tier_has_all_fields(self):
        m = ModelTier(
            name="gpt-4o", tier="medium", provider="openai",
            cost_per_m_input=2.50, cost_per_m_output=10.00,
            speed_score=1.5, quality_score=0.9, max_context=128_000,
        )
        assert m.provider == "openai"
        assert m.cost_per_m_input == 2.50
        assert m.cost_per_m_output == 10.00
