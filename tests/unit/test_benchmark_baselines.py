"""Tests for cloud comparison baselines (llmstack.benchmark.baselines)."""

from __future__ import annotations

import pytest

from llmstack.benchmark.baselines import (
    CLOUD_BASELINES,
    DEFAULT_BASELINE,
    CloudBaseline,
    get_baseline,
)
from llmstack.core.pricing import get_token_price


def test_catalog_self_keyed_and_non_empty() -> None:
    assert CLOUD_BASELINES
    for key, b in CLOUD_BASELINES.items():
        assert key == b.key


def test_every_baseline_model_is_priced() -> None:
    for b in CLOUD_BASELINES.values():
        assert get_token_price(b.model).model == b.model


def test_all_cloud_baselines_send_offdevice() -> None:
    assert all(b.sends_prompt_offdevice for b in CLOUD_BASELINES.values())


def test_cost_matches_pricing() -> None:
    b = get_baseline("gpt-4o")
    expected = get_token_price("gpt-4o").cost_usd(1000, 500)
    assert b.cost_usd(1000, 500) == pytest.approx(expected)


def test_default_baseline_lookup() -> None:
    assert get_baseline().key == DEFAULT_BASELINE
    assert get_baseline("gpt-4o").key == "gpt-4o"


def test_unknown_baseline_raises() -> None:
    with pytest.raises(KeyError):
        get_baseline("nope")


def test_price_property() -> None:
    b = CloudBaseline("k", "n", "gpt-4o-mini")
    assert b.price.model == "gpt-4o-mini"
