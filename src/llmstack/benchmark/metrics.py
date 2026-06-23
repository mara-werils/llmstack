"""Pure latency and throughput statistics for benchmark runs.

No timing or I/O happens here — these functions take already-measured numbers and
summarise them. Keeping the math pure makes the percentiles trivially testable and
the benchmark report deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass


def percentile(values: list[float], q: float) -> float:
    """Return the ``q``-th percentile (0–100) using linear interpolation.

    Matches the common "linear" method (numpy's default): the value at the
    fractional rank ``q/100 * (n - 1)``, interpolated between neighbours.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics for a set of per-request latencies (milliseconds)."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """Summarise a list of latencies (ms) into mean/min/max and percentiles."""
    if not latencies_ms:
        raise ValueError("latency_stats() requires at least one latency")
    return LatencyStats(
        count=len(latencies_ms),
        mean_ms=sum(latencies_ms) / len(latencies_ms),
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
    )


@dataclass(frozen=True)
class ThroughputStats:
    """Aggregate generation throughput across a run."""

    total_output_tokens: int
    total_time_s: float
    tokens_per_second: float
    mean_ttft_ms: float


def throughput_stats(
    output_tokens: list[int],
    times_s: list[float],
    ttfts_ms: list[float],
) -> ThroughputStats:
    """Aggregate per-request token counts, durations, and TTFTs into throughput."""
    if not output_tokens:
        raise ValueError("throughput_stats() requires at least one sample")
    total_tokens = sum(output_tokens)
    total_time = sum(times_s)
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_ttft = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else 0.0
    return ThroughputStats(
        total_output_tokens=total_tokens,
        total_time_s=total_time,
        tokens_per_second=tps,
        mean_ttft_ms=mean_ttft,
    )
