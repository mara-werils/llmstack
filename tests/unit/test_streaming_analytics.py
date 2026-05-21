"""Tests for streaming analytics."""

import time

import pytest

from llmstack.gateway.streaming_analytics import StreamingTracker, StreamMetrics


@pytest.fixture
def tracker():
    return StreamingTracker()


class TestStreamMetrics:
    def test_tokens_per_second(self):
        m = StreamMetrics(token_count=100, total_duration_ms=1000)
        assert m.tokens_per_second == 100.0

    def test_zero_duration(self):
        m = StreamMetrics(token_count=10, total_duration_ms=0)
        assert m.tokens_per_second == 0.0

    def test_avg_inter_token(self):
        m = StreamMetrics(inter_token_latencies=[10, 20, 30])
        assert m.avg_inter_token_ms == 20.0

    def test_to_dict(self):
        m = StreamMetrics(
            request_id="test", model="llama3.2",
            ttft_ms=50, token_count=10,
        )
        d = m.to_dict()
        assert d["model"] == "llama3.2"
        assert d["ttft_ms"] == 50.0


class TestStreamingTracker:
    def test_start_end_stream(self, tracker):
        tracker.start_stream("req1", model="test")
        tracker.record_chunk("req1")
        tracker.record_chunk("req1")
        metrics = tracker.end_stream("req1")

        assert metrics is not None
        assert metrics.token_count == 2
        assert metrics.chunk_count == 2
        assert metrics.ttft_ms > 0

    def test_end_unknown_stream(self, tracker):
        assert tracker.end_stream("unknown") is None

    def test_record_chunk_unknown(self, tracker):
        tracker.record_chunk("unknown")  # Should not raise

    def test_inter_token_latency(self, tracker):
        tracker.start_stream("req1")
        tracker.record_chunk("req1")
        time.sleep(0.01)
        tracker.record_chunk("req1")
        time.sleep(0.01)
        tracker.record_chunk("req1")
        metrics = tracker.end_stream("req1")

        assert len(metrics.inter_token_latencies) == 2
        assert all(itl > 0 for itl in metrics.inter_token_latencies)

    def test_summary(self, tracker):
        for i in range(5):
            tracker.start_stream(f"req{i}", model="llama3.2")
            tracker.record_chunk(f"req{i}", token_count=10)
            tracker.end_stream(f"req{i}")

        summary = tracker.get_summary()
        assert summary["total_streams"] == 5
        assert "llama3.2" in summary["by_model"]

    def test_summary_by_model(self, tracker):
        tracker.start_stream("a", model="gpt-4o")
        tracker.record_chunk("a")
        tracker.end_stream("a")

        tracker.start_stream("b", model="llama3.2")
        tracker.record_chunk("b")
        tracker.end_stream("b")

        summary = tracker.get_summary(model="gpt-4o")
        assert summary["total_streams"] == 1

    def test_get_recent(self, tracker):
        for i in range(3):
            tracker.start_stream(f"r{i}", model="test")
            tracker.record_chunk(f"r{i}")
            tracker.end_stream(f"r{i}")

        recent = tracker.get_recent(limit=2)
        assert len(recent) == 2

    def test_active_streams_count(self, tracker):
        tracker.start_stream("active1")
        tracker.start_stream("active2")
        summary = tracker.get_summary()
        assert summary["active_streams"] == 2
