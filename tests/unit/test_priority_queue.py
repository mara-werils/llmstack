"""Tests for request priority queue."""

import pytest

from llmstack.gateway.priority_queue import (
    Priority,
    PrioritizedRequest,
    QueueFullError,
    RequestPriorityQueue,
    resolve_priority,
)


@pytest.fixture
def queue():
    return RequestPriorityQueue(max_size=10)


class TestRequestPriorityQueue:
    def test_enqueue_dequeue(self, queue):
        queue.enqueue({"model": "test"}, priority=Priority.NORMAL)
        req = queue.dequeue()
        assert req is not None
        assert req.payload["model"] == "test"

    def test_priority_ordering(self, queue):
        queue.enqueue({"id": "low"}, priority=Priority.LOW)
        queue.enqueue({"id": "high"}, priority=Priority.HIGH)
        queue.enqueue({"id": "critical"}, priority=Priority.CRITICAL)

        first = queue.dequeue()
        assert first.payload["id"] == "critical"
        second = queue.dequeue()
        assert second.payload["id"] == "high"
        third = queue.dequeue()
        assert third.payload["id"] == "low"

    def test_tier_based_priority(self, queue):
        queue.enqueue({}, tier="free")  # LOW
        queue.enqueue({}, tier="enterprise")  # CRITICAL

        first = queue.dequeue()
        assert first.tier == "enterprise"

    def test_queue_full(self):
        small_queue = RequestPriorityQueue(max_size=2)
        small_queue.enqueue({})
        small_queue.enqueue({})
        with pytest.raises(QueueFullError):
            small_queue.enqueue({})

    def test_empty_dequeue(self, queue):
        assert queue.dequeue() is None

    def test_peek(self, queue):
        queue.enqueue({"id": "test"}, priority=Priority.HIGH)
        peeked = queue.peek()
        assert peeked is not None
        assert peeked.payload["id"] == "test"
        assert queue.size == 1  # Not removed

    def test_size(self, queue):
        assert queue.is_empty is True
        queue.enqueue({})
        assert queue.size == 1
        assert queue.is_empty is False

    def test_stats(self, queue):
        queue.enqueue({}, tier="pro")
        queue.enqueue({}, tier="free")
        stats = queue.get_stats()
        assert stats["queue_size"] == 2
        assert stats["total_enqueued"] == 2
        assert "pro" in stats["by_tier"]

    def test_rejected_count(self):
        tiny = RequestPriorityQueue(max_size=1)
        tiny.enqueue({})
        try:
            tiny.enqueue({})
        except QueueFullError:
            pass
        stats = tiny.get_stats()
        assert stats["total_rejected"] == 1

    def test_drop_rate(self):
        tiny = RequestPriorityQueue(max_size=1)
        tiny.enqueue({})
        try:
            tiny.enqueue({})
        except QueueFullError:
            pass
        assert tiny.drop_rate == 0.5

    def test_drop_rate_with_no_requests(self, queue):
        assert queue.drop_rate == 0.0

    def test_throughput(self, queue):
        queue.enqueue({})
        queue.enqueue({})
        queue.dequeue()
        assert queue.throughput == 1

    def test_capacity_pct(self):
        small = RequestPriorityQueue(max_size=4)
        small.enqueue({})
        assert small.capacity_pct == 25.0

    def test_clear(self, queue):
        queue.enqueue({})
        queue.enqueue({})
        removed = queue.clear()
        assert removed == 2
        assert queue.size == 0


class TestPrioritizedRequest:
    def test_auto_timestamp_when_falsy(self):
        req = PrioritizedRequest(priority=10, timestamp=0)
        assert req.timestamp > 0


class TestResolvePriority:
    def test_header_priority(self):
        assert resolve_priority({"x-priority": "high"}) == Priority.HIGH

    def test_tier_priority(self):
        assert resolve_priority({}, tier="enterprise") == Priority.CRITICAL

    def test_default_priority(self):
        assert resolve_priority({}) == Priority.NORMAL

    def test_header_overrides_tier(self):
        p = resolve_priority({"x-priority": "low"}, tier="enterprise")
        assert p == Priority.LOW

    def test_unknown_header_falls_through(self):
        p = resolve_priority({"x-priority": "unknown"}, tier="pro")
        assert p == Priority.HIGH
