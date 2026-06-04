"""Tests for analytics tracker."""

import time
import pytest
from llmstack.analytics.tracker import AnalyticsTracker, UsageEvent


@pytest.fixture
def tracker(tmp_path):
    return AnalyticsTracker(db_path=tmp_path / "test_analytics.db")


def test_track_event(tracker):
    event = UsageEvent(
        command="chat",
        model="llama3.2",
        tokens_in=100,
        tokens_out=200,
        duration=1.5,
        success=True,
        timestamp=time.time(),
    )
    tracker.track(event)

    summary = tracker.get_summary(days=1)
    assert summary["total_requests"] == 1
    assert summary["tokens_in"] == 100
    assert summary["tokens_out"] == 200


def test_summary_by_command(tracker):
    now = time.time()
    for cmd in ["chat", "chat", "review", "ask"]:
        tracker.track(
            UsageEvent(
                command=cmd,
                model="llama3.2",
                tokens_in=50,
                tokens_out=100,
                duration=1.0,
                success=True,
                timestamp=now,
            )
        )

    summary = tracker.get_summary(days=1)
    cmd_counts = {c["command"]: c["count"] for c in summary["by_command"]}
    assert cmd_counts["chat"] == 2
    assert cmd_counts["review"] == 1
    assert cmd_counts["ask"] == 1


def test_summary_by_model(tracker):
    now = time.time()
    for model in ["llama3.2", "llama3.2", "mistral"]:
        tracker.track(
            UsageEvent(
                command="chat",
                model=model,
                tokens_in=50,
                tokens_out=100,
                duration=1.0,
                success=True,
                timestamp=now,
            )
        )

    summary = tracker.get_summary(days=1)
    model_counts = {m["model"]: m["count"] for m in summary["by_model"]}
    assert model_counts["llama3.2"] == 2
    assert model_counts["mistral"] == 1


def test_success_rate(tracker):
    now = time.time()
    tracker.track(
        UsageEvent(
            command="a",
            model="m",
            tokens_in=0,
            tokens_out=0,
            duration=1.0,
            success=True,
            timestamp=now,
        )
    )
    tracker.track(
        UsageEvent(
            command="b",
            model="m",
            tokens_in=0,
            tokens_out=0,
            duration=1.0,
            success=True,
            timestamp=now,
        )
    )
    tracker.track(
        UsageEvent(
            command="c",
            model="m",
            tokens_in=0,
            tokens_out=0,
            duration=1.0,
            success=False,
            timestamp=now,
        )
    )

    summary = tracker.get_summary(days=1)
    assert abs(summary["success_rate"] - 66.7) < 1.0


def test_period_filtering(tracker):
    old = time.time() - 86400 * 60  # 60 days ago
    now = time.time()

    tracker.track(
        UsageEvent(
            command="old",
            model="m",
            tokens_in=0,
            tokens_out=0,
            duration=1.0,
            success=True,
            timestamp=old,
        )
    )
    tracker.track(
        UsageEvent(
            command="new",
            model="m",
            tokens_in=0,
            tokens_out=0,
            duration=1.0,
            success=True,
            timestamp=now,
        )
    )

    summary_7d = tracker.get_summary(days=7)
    assert summary_7d["total_requests"] == 1

    summary_90d = tracker.get_summary(days=90)
    assert summary_90d["total_requests"] == 2


def test_empty_summary(tracker):
    summary = tracker.get_summary(days=30)
    assert summary["total_requests"] == 0
    assert summary["total_tokens"] == 0
    assert summary["success_rate"] == 0
