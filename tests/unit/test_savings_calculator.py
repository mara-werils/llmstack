"""Tests for the pure savings calculator (llmstack.core.savings)."""

from __future__ import annotations

import pytest

from llmstack.core.pricing import baseline_subscription, baseline_token_price
from llmstack.core.savings import SavingsCalculator, SavingsEstimate


def test_estimate_matches_baseline_cost() -> None:
    calc = SavingsCalculator()
    est = calc.estimate(1000, 500)
    expected = baseline_token_price().cost_usd(1000, 500)
    assert est.saved_usd == pytest.approx(expected)
    assert est.cloud_cost_usd == pytest.approx(expected)
    assert est.local_cost_usd == 0.0
    assert est.baseline_model == baseline_token_price().model


def test_estimate_with_explicit_baseline() -> None:
    calc = SavingsCalculator("gpt-4o")
    est = calc.estimate(1_000_000, 0)
    # gpt-4o input is $2.50 / 1M
    assert est.saved_usd == pytest.approx(2.50)
    assert est.baseline_model == "gpt-4o"


def test_local_cost_is_subtracted_and_clamped() -> None:
    calc = SavingsCalculator("gpt-4o-mini")
    cloud = baseline_token_price("gpt-4o-mini").cost_usd(1000, 1000)
    # A local cost below cloud reduces the saving.
    est = calc.estimate(1000, 1000, local_cost_usd=cloud / 2)
    assert est.saved_usd == pytest.approx(cloud / 2)
    # A local cost above cloud clamps the saving to zero (never negative).
    est2 = calc.estimate(1000, 1000, local_cost_usd=cloud * 10)
    assert est2.saved_usd == 0.0


def test_zero_tokens_zero_saving() -> None:
    assert SavingsCalculator().estimate(0, 0).saved_usd == 0.0


def test_negative_tokens_rejected() -> None:
    calc = SavingsCalculator()
    with pytest.raises(ValueError):
        calc.estimate(-1, 0)
    with pytest.raises(ValueError):
        calc.estimate(0, -5)


def test_negative_local_cost_rejected() -> None:
    # A negative local cost would inflate the saving above the real cloud cost.
    calc = SavingsCalculator()
    with pytest.raises(ValueError):
        calc.estimate(1000, 1000, local_cost_usd=-0.01)


def test_estimate_as_dict_roundtrips_fields() -> None:
    est = SavingsCalculator().estimate(10, 20)
    d = est.as_dict()
    assert d["input_tokens"] == 10
    assert d["output_tokens"] == 20
    assert set(d) == {
        "input_tokens",
        "output_tokens",
        "baseline_model",
        "cloud_cost_usd",
        "local_cost_usd",
        "saved_usd",
    }


def test_subscription_months_covered() -> None:
    calc = SavingsCalculator()
    plan = baseline_subscription("cursor-pro")
    months = calc.subscription_months_covered(plan.effective_monthly_usd * 3, "cursor-pro")
    assert months == pytest.approx(3.0)


def test_calculator_exposes_baseline() -> None:
    calc = SavingsCalculator("gpt-4o")
    assert isinstance(calc.baseline, type(baseline_token_price("gpt-4o")))
    assert calc.baseline.model == "gpt-4o"


def test_estimate_is_frozen() -> None:
    est = SavingsCalculator().estimate(1, 1)
    assert isinstance(est, SavingsEstimate)
    with pytest.raises(Exception):
        est.saved_usd = 1.0  # type: ignore[misc]
