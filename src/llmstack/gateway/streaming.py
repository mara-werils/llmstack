"""Enhanced streaming support — SSE, WebSocket, and streaming analytics."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class StreamMetrics:
    """Metrics collected during a streaming response."""
    first_token_ms: float = 0
    total_tokens: int = 0
    total_duration_ms: float = 0
    tokens_per_second: float = 0
    chunks_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0


@dataclass
class StreamChunk:
    """A single chunk in a stream."""
    content: str
    token_index: int
    timestamp: float
    finish_reason: str | None = None
    model: str = ""


class StreamProcessor:
    """Process and enhance streaming responses."""

    def __init__(self, model: str = "", request_id: str = ""):
        self.model = model
        self.request_id = request_id
        self.metrics = StreamMetrics()
        self._start_time: float = 0
        self._first_token_time: float = 0

    async def process_stream(
        self,
        source: AsyncIterator[str],
        format: str = "sse",
    ) -> AsyncIterator[str]:
        """Process a raw token stream into formatted output."""
        self._start_time = time.time()
        token_index = 0

        async for token in source:
            if token_index == 0:
                self._first_token_time = time.time()
                self.metrics.first_token_ms = (self._first_token_time - self._start_time) * 1000

            token_index += 1
            self.metrics.total_tokens = token_index
            self.metrics.chunks_sent += 1

            chunk = StreamChunk(
                content=token,
                token_index=token_index,
                timestamp=time.time(),
                model=self.model,
            )

            if format == "sse":
                formatted = self._format_sse(chunk)
            elif format == "ndjson":
                formatted = self._format_ndjson(chunk)
            else:
                formatted = token

            self.metrics.bytes_sent += len(formatted.encode())
            yield formatted

        # Send final metrics
        self._finalize_metrics()

        if format == "sse":
            yield self._format_sse_done()
        elif format == "ndjson":
            yield self._format_ndjson_done()

    def _format_sse(self, chunk: StreamChunk) -> str:
        """Format as Server-Sent Event (OpenAI-compatible)."""
        data = {
            "id": f"chatcmpl-{self.request_id}",
            "object": "chat.completion.chunk",
            "created": int(chunk.timestamp),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk.content},
                "finish_reason": chunk.finish_reason,
            }],
        }
        return f"data: {json.dumps(data)}\n\n"

    def _format_sse_done(self) -> str:
        """Format SSE done event with metrics."""
        return (
            f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            f"data: [DONE]\n\n"
        )

    def _format_ndjson(self, chunk: StreamChunk) -> str:
        """Format as newline-delimited JSON."""
        return json.dumps({
            "token": chunk.content,
            "index": chunk.token_index,
            "model": self.model,
        }) + "\n"

    def _format_ndjson_done(self) -> str:
        """Format NDJSON done event."""
        return json.dumps({
            "done": True,
            "metrics": {
                "first_token_ms": round(self.metrics.first_token_ms, 1),
                "total_tokens": self.metrics.total_tokens,
                "tokens_per_second": round(self.metrics.tokens_per_second, 1),
                "total_duration_ms": round(self.metrics.total_duration_ms, 1),
            },
        }) + "\n"

    def _finalize_metrics(self) -> None:
        """Calculate final metrics."""
        duration = time.time() - self._start_time
        self.metrics.total_duration_ms = duration * 1000
        if duration > 0:
            self.metrics.tokens_per_second = self.metrics.total_tokens / duration


class StreamBuffer:
    """Buffer for accumulating streamed tokens with word boundary detection."""

    def __init__(self, flush_interval: float = 0.05, min_buffer_size: int = 3):
        self.buffer = ""
        self.flush_interval = flush_interval
        self.min_buffer_size = min_buffer_size
        self._last_flush = time.time()

    def add(self, token: str) -> str | None:
        """Add a token. Returns buffered text when ready to flush."""
        self.buffer += token
        now = time.time()

        # Flush on word boundaries, sentence endings, or time interval
        should_flush = (
            len(self.buffer) >= self.min_buffer_size and (
                self.buffer.endswith((" ", "\n", ".", "!", "?", ",", ";", ":"))
                or now - self._last_flush > self.flush_interval
            )
        )

        if should_flush:
            result = self.buffer
            self.buffer = ""
            self._last_flush = now
            return result

        return None

    def flush(self) -> str:
        """Force flush remaining buffer."""
        result = self.buffer
        self.buffer = ""
        return result


class StreamMultiplexer:
    """Multiplex a single stream to multiple consumers."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._history: list[str] = []

    def subscribe(self) -> asyncio.Queue:
        """Add a new subscriber."""
        queue: asyncio.Queue = asyncio.Queue()
        # Send history to catch up
        for item in self._history:
            queue.put_nowait(item)
        self._subscribers.append(queue)
        return queue

    async def publish(self, data: str) -> None:
        """Publish data to all subscribers."""
        self._history.append(data)
        for queue in self._subscribers:
            await queue.put(data)

    async def close(self) -> None:
        """Signal end of stream to all subscribers."""
        for queue in self._subscribers:
            await queue.put(None)  # Sentinel
