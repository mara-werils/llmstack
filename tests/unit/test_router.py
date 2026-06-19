"""Comprehensive tests for the Smart Model Router.

Covers the classifier, router, stats tracker, tier boundaries,
override behaviour, and all routing strategies.
"""

from __future__ import annotations

import pytest

from llmstack.gateway.router.classifier import QueryClassifier, QueryProfile
from llmstack.gateway.router.router import ModelRouter, ModelTier, RoutingDecision
from llmstack.gateway.router.stats import RouterStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier():
    return QueryClassifier()


@pytest.fixture
def sample_models():
    return [
        ModelTier(
            name="llama3.2:1b", tier="simple", max_context=8192, speed_score=3.0, quality_score=0.6
        ),
        ModelTier(
            name="llama3.2", tier="medium", max_context=8192, speed_score=1.5, quality_score=0.85
        ),
        ModelTier(
            name="llama3.1:70b",
            tier="complex",
            max_context=16384,
            speed_score=0.3,
            quality_score=1.0,
        ),
    ]


@pytest.fixture
def cost_router(sample_models):
    return ModelRouter(models=sample_models, strategy="cost")


@pytest.fixture
def quality_router(sample_models):
    return ModelRouter(models=sample_models, strategy="quality")


@pytest.fixture
def balanced_router(sample_models):
    return ModelRouter(models=sample_models, strategy="balanced")


@pytest.fixture
def latency_router(sample_models):
    return ModelRouter(models=sample_models, strategy="latency")


@pytest.fixture
def stats():
    s = RouterStats()
    s.set_largest_model("llama3.1:70b")
    return s


def _msgs(content: str, role: str = "user") -> list[dict]:
    """Helper to build a single-message conversation."""
    return [{"role": role, "content": content}]


def _conversation(*contents: str) -> list[dict]:
    """Build alternating user/assistant conversation."""
    msgs = []
    for i, c in enumerate(contents):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": c})
    return msgs


# ===================================================================
# Classifier tests
# ===================================================================


class TestClassifierSimple:
    """Simple queries should score < 0.35."""

    def test_hello(self, classifier):
        p = classifier.classify(_msgs("hello"))
        assert p.tier == "simple"
        assert p.score < 0.35

    def test_hi(self, classifier):
        p = classifier.classify(_msgs("hi"))
        assert p.tier == "simple"
        assert p.score < 0.35

    def test_thanks(self, classifier):
        p = classifier.classify(_msgs("thanks"))
        assert p.tier == "simple"
        assert p.score < 0.35

    def test_simple_question(self, classifier):
        p = classifier.classify(_msgs("What is the capital of France?"))
        assert p.tier == "simple"
        assert p.score < 0.35

    def test_translate_short(self, classifier):
        p = classifier.classify(_msgs("Translate 'hello' to Spanish"))
        assert p.tier == "simple"

    def test_empty_messages(self, classifier):
        p = classifier.classify([])
        assert p.tier == "simple"
        assert p.score == 0.0


class TestClassifierMedium:
    """Medium queries should score >= 0.35 and < 0.7."""

    def test_explain_concept(self, classifier):
        p = classifier.classify(_msgs("Explain how neural networks learn through backpropagation"))
        assert p.tier == "medium", f"Expected medium, got {p.tier} (score={p.score})"

    def test_compare_things(self, classifier):
        p = classifier.classify(
            _msgs("Compare and contrast REST and GraphQL APIs for building web services")
        )
        assert p.tier in ("medium", "complex"), f"Expected medium+, got {p.tier} (score={p.score})"

    def test_debug_request(self, classifier):
        p = classifier.classify(
            _msgs(
                "Debug this Python function that is supposed to sort a list but returns incorrect results"
            )
        )
        assert p.tier in ("medium", "complex")
        assert p.score >= 0.35


class TestClassifierComplex:
    """Complex queries should score >= 0.7."""

    def test_implement_algorithm(self, classifier):
        p = classifier.classify(
            _msgs(
                "Implement a distributed consensus algorithm similar to Raft in Python, "
                "including leader election, log replication, and fault tolerance handling "
                "for up to 5 nodes. Design the architecture to be production-ready."
            )
        )
        assert p.tier == "complex", f"Expected complex, got {p.tier} (score={p.score})"

    def test_architect_system(self, classifier):
        p = classifier.classify(
            _msgs(
                "Design a scalable microservice architecture for a real-time trading platform "
                "that handles high-availability requirements, considering fault tolerance, "
                "given strict latency constraints and also regulatory compliance."
            )
        )
        assert p.tier == "complex", f"Expected complex, got {p.tier} (score={p.score})"


class TestClassifierCodeDetection:
    """Queries with code should score higher."""

    def test_code_block(self, classifier):
        msg = (
            "What's wrong with this code?\n"
            "```python\n"
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n"
            "```"
        )
        p = classifier.classify(_msgs(msg))
        assert p.factors["code_detection"] > 0.3

    def test_inline_code_keywords(self, classifier):
        msg = "The function def process_data returns None when import pandas fails"
        p = classifier.classify(_msgs(msg))
        assert p.factors["code_detection"] > 0

    def test_programming_terms(self, classifier):
        msg = "Explain how to use a decorator with a closure in Python for caching with recursion"
        p = classifier.classify(_msgs(msg))
        assert p.factors["code_detection"] > 0


class TestClassifierConversationDepth:
    """Deeper conversations should have higher depth scores."""

    def test_single_message(self, classifier):
        p = classifier.classify(_msgs("Hello"))
        assert p.factors["conversation_depth"] == 0.1

    def test_deep_conversation(self, classifier):
        msgs = _conversation(
            "Tell me about Python",
            "Python is a programming language...",
            "How does the GIL work?",
            "The Global Interpreter Lock...",
            "Can we work around it?",
            "Yes, you can use multiprocessing...",
            "What about asyncio?",
            "asyncio uses cooperative multitasking...",
            "Compare the performance implications",
        )
        p = classifier.classify(msgs)
        assert p.factors["conversation_depth"] >= 0.5


class TestClassifierSystemPrompt:
    """Long system prompts should increase complexity."""

    def test_long_system_prompt(self, classifier):
        system = " ".join(["You are an expert software architect."] * 50)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": "How should I design this?"},
        ]
        p = classifier.classify(msgs)
        assert p.factors["system_prompt"] >= 0.35


class TestClassifierLanguageMix:
    """Mixed-language queries should get a bump."""

    def test_multilingual(self, classifier):
        p = classifier.classify(_msgs("Translate this to Japanese: hello world"))
        # No non-Latin chars, so no bump
        assert p.factors["language_mix"] == 0.0

    def test_mixed_script(self, classifier):
        p = classifier.classify(
            _msgs(
                "Explain the concept of recursion. Also explain: \u0420\u0435\u043a\u0443\u0440\u0441\u0438\u044f"
            )
        )
        assert p.factors["language_mix"] > 0


class TestClassifierTierBoundaries:
    """Test edge cases around tier boundaries."""

    def test_score_clamped_to_01(self, classifier):
        p = classifier.classify(_msgs("x"))
        assert 0.0 <= p.score <= 1.0

    def test_profile_dataclass_validation(self):
        # Valid
        p = QueryProfile(score=0.5, tier="medium", factors={})
        assert p.tier == "medium"

        # Score clamping
        p = QueryProfile(score=1.5, tier="complex", factors={})
        assert p.score == 1.0

        p = QueryProfile(score=-0.3, tier="simple", factors={})
        assert p.score == 0.0

    def test_invalid_tier_rejected(self):
        with pytest.raises(ValueError):
            QueryProfile(score=0.5, tier="invalid", factors={})


# ===================================================================
# Router tests
# ===================================================================


class TestRouterCostStrategy:
    """Cost strategy should pick the smallest adequate model."""

    def test_simple_query_picks_small(self, cost_router):
        d = cost_router.route(_msgs("hello"))
        assert d.model == "llama3.2:1b"
        assert d.profile.tier == "simple"

    def test_complex_query_picks_large(self, cost_router):
        d = cost_router.route(
            _msgs(
                "Implement a production-ready distributed consensus algorithm with fault tolerance"
            )
        )
        assert d.model == "llama3.1:70b"


class TestRouterQualityStrategy:
    """Quality strategy should pick the best model for the tier."""

    def test_simple_picks_best_available(self, quality_router):
        d = quality_router.route(_msgs("hello"))
        # Even for simple, quality picks the highest quality among adequate (simple+)
        assert d.model in ("llama3.2:1b", "llama3.2", "llama3.1:70b")
        # It should be the highest quality among models with tier >= simple
        assert d.model == "llama3.1:70b"  # highest quality_score

    def test_complex_picks_best(self, quality_router):
        d = quality_router.route(_msgs("Architect a scalable end-to-end distributed system"))
        assert d.model == "llama3.1:70b"


class TestRouterBalancedStrategy:
    def test_medium_query(self, balanced_router):
        d = balanced_router.route(_msgs("Explain how garbage collection works in Java in detail"))
        # Balanced should consider both quality and speed
        assert d.model in ("llama3.2", "llama3.2:1b", "llama3.1:70b")


class TestRouterLatencyStrategy:
    def test_picks_fastest_adequate(self, latency_router):
        d = latency_router.route(_msgs("hello"))
        # For simple tier, should pick fastest with tier >= simple
        assert d.model == "llama3.2:1b"  # speed_score=3.0 is highest


class TestRouterOverride:
    """Model override should bypass classification."""

    def test_override_forces_model(self, cost_router):
        cost_router.override("llama3.1:70b")
        d = cost_router.route(_msgs("hello"))
        assert d.model == "llama3.1:70b"
        # Clean up
        cost_router.override(None)

    def test_override_clear(self, cost_router):
        cost_router.override("llama3.1:70b")
        cost_router.override(None)
        d = cost_router.route(_msgs("hello"))
        assert d.model == "llama3.2:1b"  # back to normal routing


class TestRouterDecision:
    """RoutingDecision should contain alternatives and speedup."""

    def test_alternatives_present(self, cost_router):
        d = cost_router.route(_msgs("hello"))
        assert len(d.alternatives) == 2
        assert d.model not in d.alternatives

    def test_speedup_calculated(self, cost_router):
        d = cost_router.route(_msgs("hello"))
        # small model vs 70b should show speedup
        assert d.estimated_speedup > 1.0


class TestRouterValidation:
    """Router should reject invalid configs."""

    def test_invalid_strategy_rejected(self, sample_models):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ModelRouter(models=sample_models, strategy="turbo")

    def test_empty_models_rejected(self):
        with pytest.raises(ValueError, match="At least one"):
            ModelRouter(models=[], strategy="cost")


# ===================================================================
# Stats tests
# ===================================================================


class TestRouterStats:
    def test_record_and_summary(self, stats):
        profile = QueryProfile(score=0.2, tier="simple", factors={})
        decision = RoutingDecision(model="llama3.2:1b", profile=profile)
        stats.record(decision, latency_ms=50.0)

        s = stats.summary()
        assert s["total_requests"] == 1
        assert "llama3.2:1b" in s["model_distribution"]
        assert "simple" in s["tier_distribution"]

    def test_negative_cost_clamped_to_zero(self, stats):
        profile = QueryProfile(score=0.2, tier="simple", factors={})
        decision = RoutingDecision(model="llama3.2:1b", profile=profile)
        stats.record(decision, latency_ms=10.0, cost_usd=-5.0)
        s = stats.summary()
        assert s["total_cost_usd"] == 0.0
        assert all(c >= 0.0 for c in s["cost_by_provider_usd"].values())

    def test_savings_tracking(self, stats):
        # Two requests to small model (avoided large)
        for _ in range(2):
            p = QueryProfile(score=0.1, tier="simple", factors={})
            d = RoutingDecision(model="llama3.2:1b", profile=p)
            stats.record(d, latency_ms=30.0)

        # One request to large model
        p = QueryProfile(score=0.9, tier="complex", factors={})
        d = RoutingDecision(model="llama3.1:70b", profile=p)
        stats.record(d, latency_ms=500.0)

        s = stats.summary()
        assert s["total_requests"] == 3
        # 2 out of 3 avoided the large model
        assert s["estimated_savings_pct"] == pytest.approx(66.7, abs=0.1)

    def test_latency_tracking(self, stats):
        for lat in [10.0, 20.0, 30.0]:
            p = QueryProfile(score=0.1, tier="simple", factors={})
            d = RoutingDecision(model="llama3.2:1b", profile=p)
            stats.record(d, latency_ms=lat)

        s = stats.summary()
        assert s["avg_latency_by_model_ms"]["llama3.2:1b"] == pytest.approx(20.0)
        assert s["avg_latency_by_tier_ms"]["simple"] == pytest.approx(20.0)

    def test_recent_decisions_capped(self, stats):
        for i in range(25):
            p = QueryProfile(score=0.1, tier="simple", factors={})
            d = RoutingDecision(model="llama3.2:1b", profile=p)
            stats.record(d, latency_ms=float(i))

        s = stats.summary()
        # summary returns last 20
        assert len(s["recent_decisions"]) == 20

    def test_empty_summary(self):
        s = RouterStats()
        result = s.summary()
        assert result["total_requests"] == 0
        assert result["estimated_savings_pct"] == 0.0

    def test_reset_clears_all_counters(self, stats):
        p = QueryProfile(score=0.1, tier="simple", factors={})
        d = RoutingDecision(model="llama3.2:1b", profile=p)
        stats.record(d, latency_ms=10.0, cost_usd=0.01)

        stats.reset()

        s = stats.summary()
        assert s["total_requests"] == 0
        assert s["model_distribution"] == {}
        assert s["recent_decisions"] == []
        assert s["total_cost_usd"] == 0.0

    def test_provider_and_cost_tracking(self, stats):
        p = QueryProfile(score=0.1, tier="simple", factors={})
        d = RoutingDecision(model="llama3.2:1b", profile=p, provider="openai")
        stats.record(d, latency_ms=10.0, cost_usd=0.05)

        s = stats.summary()
        assert s["provider_distribution"]["openai"]["count"] == 1
        assert s["cost_by_provider_usd"]["openai"] == 0.05
        assert s["total_cost_usd"] == 0.05
        assert s["recent_decisions"][0]["provider"] == "openai"
        assert s["recent_decisions"][0]["cost_usd"] == 0.05

    def test_record_defaults_provider_to_local_when_missing(self, stats):
        p = QueryProfile(score=0.1, tier="simple", factors={})
        d = RoutingDecision(model="llama3.2:1b", profile=p)
        assert not hasattr(d, "provider") or d.provider == "local"
        stats.record(d, latency_ms=10.0)
        s = stats.summary()
        assert s["provider_distribution"]["local"]["count"] == 1
