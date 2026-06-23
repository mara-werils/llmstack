"""Tests for benchmark statistics (llmstack.benchmark.metrics)."""

from __future__ import annotations

import pytest

from llmstack.benchmark.metrics import (
    LatencyStats,
    latency_stats,
    percentile,
    throughput_stats,
)


def test_percentile_known_values() -> None:
    data = [1, 2, 3, 4, 5]
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 5
    assert percentile(data, 50) == 3
    # 25th percentile: rank = 0.25 * 4 = 1.0 -> exactly data[1] = 2
    assert percentile(data, 25) == 2


def test_percentile_interpolates() -> None:
    # rank = 0.9 * 1 = 0.9 -> 10 + (20-10)*0.9 = 19
    assert percentile([10, 20], 90) == pytest.approx(19.0)


def test_percentile_single_value() -> None:
    assert percentile([7.0], 95) == 7.0


def test_percentile_unsorted_input() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3


def test_percentile_validation() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], -1)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_stats_basic() -> None:
    stats = latency_stats([10.0, 20.0, 30.0])
    assert isinstance(stats, LatencyStats)
    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.min_ms == 10.0
    assert stats.max_ms == 30.0
    assert stats.p50_ms == pytest.approx(20.0)


def test_latency_stats_requires_data() -> None:
    with pytest.raises(ValueError):
        latency_stats([])


def test_throughput_stats_basic() -> None:
    stats = throughput_stats([100, 200], [1.0, 1.0], [50.0, 150.0])
    assert stats.total_output_tokens == 300
    assert stats.total_time_s == 2.0
    assert stats.tokens_per_second == pytest.approx(150.0)
    assert stats.mean_ttft_ms == pytest.approx(100.0)


def test_throughput_zero_time_is_safe() -> None:
    stats = throughput_stats([100], [0.0], [])
    assert stats.tokens_per_second == 0.0
    assert stats.mean_ttft_ms == 0.0


def test_throughput_requires_samples() -> None:
    with pytest.raises(ValueError):
        throughput_stats([], [], [])
