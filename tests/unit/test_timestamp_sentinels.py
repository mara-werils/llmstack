"""An explicit timestamp of 0.0 must not be overwritten by __post_init__.

Several gateway dataclasses auto-stamp a creation/attempt time when one isn't
supplied. They used ``if not self.<ts>:`` to detect "unset", but 0.0 is falsy,
so a caller passing an explicit epoch-0 timestamp (deterministic tests, replay,
deserialization) had it silently replaced with time.time(). Using a None
sentinel distinguishes "not provided" from a real zero.
"""

from __future__ import annotations

from llmstack.gateway.batch import BatchJob
from llmstack.gateway.cost_tracker import BudgetAlert
from llmstack.gateway.prompt_cache import CachedPrefix
from llmstack.gateway.retry import RetryAttempt


def test_retry_attempt_preserves_explicit_zero():
    assert RetryAttempt(attempt=1, provider="p", timestamp=0.0).timestamp == 0.0
    # Unset still gets a real wall-clock timestamp.
    assert RetryAttempt(attempt=1, provider="p").timestamp > 0


def test_batch_job_preserves_explicit_zero():
    assert BatchJob(created_at=0.0).created_at == 0.0
    assert BatchJob().created_at > 0


def test_cached_prefix_preserves_explicit_zero():
    assert CachedPrefix(hash="h", prefix_text="t", created_at=0.0).created_at == 0.0
    assert CachedPrefix(hash="h", prefix_text="t").created_at > 0


def test_budget_alert_preserves_explicit_zero():
    alert = BudgetAlert(
        budget_name="b", current_spend=1.0, limit_usd=2.0, percent_used=50.0, triggered_at=0.0
    )
    assert alert.triggered_at == 0.0
    assert (
        BudgetAlert(
            budget_name="b", current_spend=1.0, limit_usd=2.0, percent_used=50.0
        ).triggered_at
        > 0
    )
