"""Request priority queue — priority-based request scheduling.

Prioritize requests based on API key tier, request type, or custom
priority headers. Higher-priority requests are processed first when
the system is under load.
"""

from __future__ import annotations

import asyncio
import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Lock


class Priority(IntEnum):
    """Request priority levels (lower number = higher priority)."""

    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 80
    BACKGROUND = 100


# Map API key tiers to priorities
TIER_PRIORITY: dict[str, Priority] = {
    "enterprise": Priority.CRITICAL,
    "pro": Priority.HIGH,
    "standard": Priority.NORMAL,
    "free": Priority.LOW,
    "batch": Priority.BACKGROUND,
}


@dataclass(order=True)
class PrioritizedRequest:
    """A request with priority ordering."""

    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False, default="")
    payload: dict = field(compare=False, default_factory=dict)
    api_key: str = field(compare=False, default="")
    tier: str = field(compare=False, default="standard")
    future: asyncio.Future | None = field(compare=False, default=None, repr=False)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()


class RequestPriorityQueue:
    """Priority queue for LLM requests with tier-based scheduling."""

    def __init__(self, max_size: int = 1000):
        self._lock = Lock()
        self._heap: list[PrioritizedRequest] = []
        self._max_size = max_size
        self._total_enqueued: int = 0
        self._total_processed: int = 0
        self._total_rejected: int = 0

    def enqueue(
        self,
        payload: dict,
        priority: int | None = None,
        api_key: str = "",
        tier: str = "standard",
    ) -> PrioritizedRequest:
        """Add a request to the priority queue.

        Returns the PrioritizedRequest with a future to await.
        Raises QueueFullError if queue is at capacity.
        """
        if priority is None:
            priority = TIER_PRIORITY.get(tier, Priority.NORMAL)

        with self._lock:
            if len(self._heap) >= self._max_size:
                self._total_rejected += 1
                raise QueueFullError(f"Queue full ({self._max_size} requests pending)")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            future = loop.create_future()

            request = PrioritizedRequest(
                priority=priority,
                timestamp=time.time(),
                payload=payload,
                api_key=api_key,
                tier=tier,
                future=future,
            )
            heapq.heappush(self._heap, request)
            self._total_enqueued += 1

        return request

    def dequeue(self) -> PrioritizedRequest | None:
        """Get the highest-priority request from the queue."""
        with self._lock:
            if not self._heap:
                return None
            request = heapq.heappop(self._heap)
            self._total_processed += 1
            return request

    def peek(self) -> PrioritizedRequest | None:
        """Peek at the highest-priority request without removing it."""
        with self._lock:
            return self._heap[0] if self._heap else None

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def capacity_pct(self) -> float:
        """Return queue usage as a percentage of max capacity."""
        return (self.size / self._max_size) * 100.0 if self._max_size > 0 else 0.0

    def clear(self) -> int:
        """Remove all pending requests from the queue. Returns count removed."""
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            return count

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            priority_counts: dict[str, int] = {}
            tier_counts: dict[str, int] = {}
            for req in self._heap:
                pname = (
                    Priority(req.priority).name
                    if req.priority in Priority._value2member_map_
                    else str(req.priority)
                )
                priority_counts[pname] = priority_counts.get(pname, 0) + 1
                tier_counts[req.tier] = tier_counts.get(req.tier, 0) + 1

            return {
                "queue_size": len(self._heap),
                "max_size": self._max_size,
                "total_enqueued": self._total_enqueued,
                "total_processed": self._total_processed,
                "total_rejected": self._total_rejected,
                "by_priority": priority_counts,
                "by_tier": tier_counts,
            }


class QueueFullError(Exception):
    """Raised when the priority queue is at capacity."""

    pass


def resolve_priority(
    headers: dict[str, str],
    api_key: str = "",
    tier: str = "",
) -> int:
    """Resolve request priority from headers, API key, or tier.

    Checks X-Priority header first, then tier mapping.
    """
    # Explicit priority header
    priority_header = headers.get("x-priority", "").lower()
    if priority_header:
        header_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "normal": Priority.NORMAL,
            "low": Priority.LOW,
            "background": Priority.BACKGROUND,
        }
        if priority_header in header_map:
            return header_map[priority_header]

    # Tier-based priority
    if tier:
        return TIER_PRIORITY.get(tier, Priority.NORMAL)

    return Priority.NORMAL
