"""Tests for local-vs-cloud comparison (llmstack.benchmark.compare)."""

from __future__ import annotations

import pytest

from llmstack.benchmark.baselines import get_baseline
from llmstack.benchmark.compare import Comparison, compare
from llmstack.benchmark.privacy import EgressProof
from llmstack.benchmark.runner import RunResult, TaskResult


def _run(input_tokens=1000, output_tokens=500) -> RunResult:
    return RunResult(
        suite_name="t",
        suite_version="1",
        model="llama3.2",
        results=(TaskResult("a", "latency", input_tokens, output_tokens, 0.5, 0.0, ok=True),),
    )


def test_cost_uses_baseline_pricing() -> None:
    cmp = compare(_run(), baseline="gpt-4o")
    expected = get_baseline("gpt-4o").cost_usd(1000, 500)
    assert isinstance(cmp, Comparison)
    assert cmp.cloud_cost_usd == pytest.approx(expected)
    assert cmp.local_cost_usd == 0.0
    assert cmp.saved_usd == pytest.approx(expected)


def test_default_baseline() -> None:
    cmp = compare(_run())
    assert cmp.baseline_key == get_baseline().key


def test_local_cost_reduces_saving_and_clamps() -> None:
    cloud = get_baseline("gpt-4o").cost_usd(1000, 500)
    cmp = compare(_run(), baseline="gpt-4o", local_cost_usd=cloud * 2)
    assert cmp.saved_usd == 0.0


def test_privacy_with_clean_egress_proof() -> None:
    proof = EgressProof(is_local_only=True, total_connections=3, external_connections=())
    cmp = compare(_run(), egress_proof=proof)
    assert cmp.local_sends_offdevice is False
    assert cmp.cloud_sends_offdevice is True


def test_privacy_with_dirty_egress_proof() -> None:
    proof = EgressProof(
        is_local_only=False, total_connections=1, external_connections=("8.8.8.8:9",)
    )
    cmp = compare(_run(), egress_proof=proof)
    assert cmp.local_sends_offdevice is True


def test_privacy_without_proof_makes_no_local_claim() -> None:
    cmp = compare(_run())
    # No proof -> we do not claim the local run was private.
    assert cmp.local_sends_offdevice is False


def test_zero_tokens_zero_cost() -> None:
    cmp = compare(_run(0, 0))
    assert cmp.cloud_cost_usd == 0.0
    assert cmp.saved_usd == 0.0


def test_as_dict_keys() -> None:
    d = compare(_run()).as_dict()
    assert "saved_usd" in d and "cloud_sends_offdevice" in d
