"""Tests for the paid-alternative pricing catalog (llmstack.core.pricing)."""

from __future__ import annotations

import pytest

from llmstack.core import pricing
from llmstack.core.pricing import (
    API_PRICING,
    DEFAULT_API_BASELINE,
    DEFAULT_SUBSCRIPTION_BASELINE,
    PRICING_AS_OF,
    SUBSCRIPTIONS,
    SubscriptionPlan,
    TokenPrice,
    baseline_subscription,
    baseline_token_price,
    cheapest_subscription,
    get_subscription,
    get_token_price,
)


def test_catalogs_are_non_empty_and_self_keyed() -> None:
    assert SUBSCRIPTIONS and API_PRICING
    for key, plan in SUBSCRIPTIONS.items():
        assert key == plan.key
    for model, price in API_PRICING.items():
        assert model == price.model


def test_every_entry_is_dated_and_sourced() -> None:
    for plan in SUBSCRIPTIONS.values():
        assert plan.source.startswith("http")
        assert plan.as_of
        assert plan.monthly_usd > 0
    for price in API_PRICING.values():
        assert price.source.startswith("http")
        assert price.as_of
        assert price.input_per_million >= 0
        assert price.output_per_million >= 0


def test_pricing_as_of_is_default_on_entries() -> None:
    # Entries that don't override as_of inherit the catalog month.
    assert get_subscription("copilot-pro").as_of == PRICING_AS_OF
    assert get_token_price("gpt-4o").as_of == PRICING_AS_OF


def test_token_price_cost_math() -> None:
    price = TokenPrice("x", "v", 2.0, 10.0, "https://example.com")
    # 1M input @ $2 + 0.5M output @ $10 = 2 + 5 = 7
    assert price.cost_usd(1_000_000, 500_000) == pytest.approx(7.0)
    assert price.cost_usd(0, 0) == 0.0


def test_effective_monthly_prefers_cheaper_annual() -> None:
    plan = SubscriptionPlan("k", "n", "v", 10.0, "https://example.com", annual_usd=100.0)
    # 100/12 = 8.33 < 10 monthly
    assert plan.effective_monthly_usd == pytest.approx(100.0 / 12.0)


def test_effective_monthly_keeps_monthly_when_annual_not_cheaper() -> None:
    plan = SubscriptionPlan("k", "n", "v", 10.0, "https://example.com", annual_usd=240.0)
    assert plan.effective_monthly_usd == 10.0


def test_effective_monthly_without_annual() -> None:
    plan = SubscriptionPlan("k", "n", "v", 19.0, "https://example.com")
    assert plan.effective_monthly_usd == 19.0


def test_lookups_raise_on_unknown() -> None:
    with pytest.raises(KeyError):
        get_subscription("does-not-exist")
    with pytest.raises(KeyError):
        get_token_price("does-not-exist")


def test_baseline_helpers_default() -> None:
    assert baseline_token_price().model == DEFAULT_API_BASELINE
    assert baseline_subscription().key == DEFAULT_SUBSCRIPTION_BASELINE


def test_baseline_helpers_explicit() -> None:
    assert baseline_token_price("gpt-4o").model == "gpt-4o"
    assert baseline_subscription("cursor-pro").key == "cursor-pro"


def test_cheapest_subscription_is_minimum() -> None:
    cheapest = cheapest_subscription()
    assert cheapest.effective_monthly_usd == min(
        p.effective_monthly_usd for p in SUBSCRIPTIONS.values()
    )


def test_default_baselines_exist_in_catalogs() -> None:
    assert pricing.DEFAULT_API_BASELINE in API_PRICING
    assert pricing.DEFAULT_SUBSCRIPTION_BASELINE in SUBSCRIPTIONS
