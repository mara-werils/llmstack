"""Request replay — record and replay LLM requests for debugging.

Captures full request/response pairs that can be replayed later
for debugging, testing, or regression checking.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class RecordedRequest:
    """A recorded request/response pair."""

    id: str = ""
    timestamp: float = 0.0

    # Request
    model: str = ""
    provider: str = ""
    messages: list[dict] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Response
    response: dict = field(default_factory=dict)
    status_code: int = 200
    error: str = ""

    # Metrics
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    # Tags for filtering
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def total_tokens(self) -> int:
        """Return combined input + output token count."""
        return self.input_tokens + self.output_tokens

    @property
    def is_error(self) -> bool:
        """Return True when the recorded request resulted in an error."""
        return bool(self.error) or self.status_code >= 400

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "provider": self.provider,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "response": self.response,
            "status_code": self.status_code,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cached": self.cached,
            "tags": self.tags,
            "notes": self.notes,
        }

    def to_replay_payload(self) -> dict:
        """Convert to a payload suitable for replaying."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_params)
        return payload


class ReplayStore:
    """In-memory store for recorded requests with export/import."""

    def __init__(self, max_size: int = 2000):
        self._lock = Lock()
        self._records: deque[RecordedRequest] = deque(maxlen=max_size)
        self._recording: bool = True

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def record(self, request: RecordedRequest) -> None:
        """Record a request/response pair."""
        if not self._recording:
            return
        with self._lock:
            self._records.append(request)

    def get(self, record_id: str) -> RecordedRequest | None:
        """Get a specific recorded request."""
        with self._lock:
            for r in self._records:
                if r.id == record_id:
                    return r
            return None

    def list_records(
        self,
        model: str | None = None,
        provider: str | None = None,
        tag: str | None = None,
        has_error: bool | None = None,
        limit: int = 50,
    ) -> list[RecordedRequest]:
        """List recorded requests with optional filters."""
        with self._lock:
            results = []
            for r in reversed(self._records):
                if model and r.model != model:
                    continue
                if provider and r.provider != provider:
                    continue
                if tag and tag not in r.tags:
                    continue
                if has_error is True and not r.error:
                    continue
                if has_error is False and r.error:
                    continue
                results.append(r)
                if len(results) >= limit:
                    break
            return results

    def clear(self) -> int:
        """Clear all records. Returns count cleared."""
        with self._lock:
            count = len(self._records)
            self._records.clear()
            return count

    def export_jsonl(self, path: str | Path) -> int:
        """Export records to JSONL file."""
        with self._lock:
            records = list(self._records)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
        return len(records)

    def import_jsonl(self, path: str | Path) -> int:
        """Import records from JSONL file."""
        path = Path(path)
        if not path.exists():
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = RecordedRequest(
                    id=data.get("id", ""),
                    timestamp=data.get("timestamp", 0),
                    model=data.get("model", ""),
                    provider=data.get("provider", ""),
                    messages=data.get("messages", []),
                    temperature=data.get("temperature", 0),
                    max_tokens=data.get("max_tokens"),
                    stream=data.get("stream", False),
                    response=data.get("response", {}),
                    status_code=data.get("status_code", 200),
                    error=data.get("error", ""),
                    latency_ms=data.get("latency_ms", 0),
                    input_tokens=data.get("input_tokens", 0),
                    output_tokens=data.get("output_tokens", 0),
                    cost_usd=data.get("cost_usd", 0),
                    cached=data.get("cached", False),
                    tags=data.get("tags", []),
                    notes=data.get("notes", ""),
                )
                with self._lock:
                    self._records.append(record)
                count += 1
        return count

    def get_stats(self) -> dict:
        """Get recording statistics."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {"total": 0, "recording": self._recording}

        errors = sum(1 for r in records if r.error)
        total_cost = sum(r.cost_usd for r in records)
        models = {}
        for r in records:
            models[r.model] = models.get(r.model, 0) + 1

        return {
            "total": len(records),
            "recording": self._recording,
            "errors": errors,
            "total_cost_usd": round(total_cost, 6),
            "models": models,
        }
