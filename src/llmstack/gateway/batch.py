"""Batch processing — process multiple LLM requests in parallel.

Supports batch chat completions with configurable concurrency,
progress tracking, and result aggregation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchRequest:
    """A single request within a batch."""

    id: str = ""
    messages: list[dict] = field(default_factory=list)
    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class BatchResult:
    """Result of a single batch request."""

    request_id: str = ""
    status: str = "pending"
    response: dict = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "response": self.response,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class BatchJob:
    """A batch processing job containing multiple requests."""

    id: str = ""
    status: BatchStatus = BatchStatus.PENDING
    requests: list[BatchRequest] = field(default_factory=list)
    results: list[BatchResult] = field(default_factory=list)
    concurrency: int = 5
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()

    @property
    def progress(self) -> float:
        if not self.requests:
            return 0.0
        completed = sum(1 for r in self.results if r.status != "pending")
        return completed / len(self.requests)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "total_requests": len(self.requests),
            "completed_requests": sum(
                1 for r in self.results if r.status in ("completed", "failed")
            ),
            "progress": round(self.progress * 100, 1),
            "concurrency": self.concurrency,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def summary(self) -> dict:
        """Detailed summary with all results."""
        d = self.to_dict()
        d["results"] = [r.to_dict() for r in self.results]
        succeeded = sum(1 for r in self.results if r.status == "completed")
        failed = sum(1 for r in self.results if r.status == "failed")
        latencies = [r.latency_ms for r in self.results if r.latency_ms > 0]
        d["succeeded"] = succeeded
        d["failed"] = failed
        d["avg_latency_ms"] = round(
            sum(latencies) / len(latencies), 1
        ) if latencies else 0.0
        return d


class BatchProcessor:
    """Manages batch processing jobs with concurrency control."""

    MAX_BATCH_SIZE = 100
    MAX_CONCURRENT_JOBS = 10

    def __init__(self):
        self._lock = Lock()
        self._jobs: dict[str, BatchJob] = {}

    def create_job(
        self,
        requests: list[dict],
        concurrency: int = 5,
        metadata: dict | None = None,
    ) -> BatchJob:
        """Create a new batch job from request payloads."""
        if len(requests) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(requests)} exceeds max {self.MAX_BATCH_SIZE}"
            )

        batch_requests = []
        for req in requests:
            batch_requests.append(BatchRequest(
                messages=req.get("messages", []),
                model=req.get("model", ""),
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens"),
                metadata=req.get("metadata", {}),
            ))

        job = BatchJob(
            requests=batch_requests,
            results=[
                BatchResult(request_id=br.id) for br in batch_requests
            ],
            concurrency=min(concurrency, 20),
            metadata=metadata or {},
        )

        with self._lock:
            self._jobs[job.id] = job

        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        """Get a batch job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[BatchJob]:
        """List recent batch jobs."""
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )
            return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (BatchStatus.COMPLETED, BatchStatus.CANCELLED):
                return False
            job.status = BatchStatus.CANCELLED
            return True

    async def execute_job(
        self,
        job_id: str,
        handler,
    ) -> BatchJob | None:
        """Execute a batch job using the provided async handler.

        handler(payload: dict) -> dict should process a single chat request.
        """
        job = self.get_job(job_id)
        if job is None or job.status != BatchStatus.PENDING:
            return None

        job.status = BatchStatus.RUNNING
        job.started_at = time.time()

        semaphore = asyncio.Semaphore(job.concurrency)

        async def _process_one(idx: int, req: BatchRequest) -> None:
            async with semaphore:
                if job.status == BatchStatus.CANCELLED:
                    return
                t0 = time.monotonic()
                try:
                    payload = {
                        "messages": req.messages,
                        "model": req.model,
                        "temperature": req.temperature,
                        "stream": False,
                    }
                    if req.max_tokens:
                        payload["max_tokens"] = req.max_tokens

                    result = await handler(payload)
                    elapsed = (time.monotonic() - t0) * 1000

                    usage = result.get("usage", {})
                    job.results[idx] = BatchResult(
                        request_id=req.id,
                        status="completed",
                        response=result,
                        latency_ms=elapsed,
                        tokens_used=usage.get("total_tokens", 0),
                        cost_usd=result.get("x_llmstack", {}).get("cost_usd", 0.0),
                    )
                except Exception as exc:
                    elapsed = (time.monotonic() - t0) * 1000
                    job.results[idx] = BatchResult(
                        request_id=req.id,
                        status="failed",
                        error=str(exc),
                        latency_ms=elapsed,
                    )

        tasks = [
            _process_one(i, req) for i, req in enumerate(job.requests)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        job.completed_at = time.time()
        if job.status != BatchStatus.CANCELLED:
            all_failed = all(r.status == "failed" for r in job.results)
            if all_failed:
                job.status = BatchStatus.FAILED
            else:
                job.status = BatchStatus.COMPLETED

        return job
