"""Tests for enhanced streaming support."""

import pytest
from llmstack.gateway.streaming import (
    StreamBuffer,
    StreamError,
    StreamMultiplexer,
    StreamProcessor,
    StreamSizeError,
    StreamTimeoutError,
)


@pytest.mark.asyncio
async def test_stream_size_cap_raises_size_error():
    async def token_source():
        for _ in range(100):
            yield "x" * 100

    # Tiny cap so the first formatted chunk trips the limit.
    processor = StreamProcessor(model="m", request_id="r", max_bytes=10)
    with pytest.raises(StreamSizeError):
        async for _ in processor.process_stream(token_source(), format="sse"):
            pass
    assert processor.metrics.errors == 1


def test_stream_errors_share_a_base():
    # Callers can catch both failure modes via the shared base, but the size
    # cap is no longer misreported as a timeout.
    assert issubclass(StreamSizeError, StreamError)
    assert issubclass(StreamTimeoutError, StreamError)
    assert not issubclass(StreamSizeError, StreamTimeoutError)


@pytest.mark.asyncio
async def test_stream_processor_sse():
    async def token_source():
        for word in ["Hello", " ", "world", "!"]:
            yield word

    processor = StreamProcessor(model="test-model", request_id="abc123")
    chunks = []
    async for chunk in processor.process_stream(token_source(), format="sse"):
        chunks.append(chunk)

    # Should have data chunks + done event
    assert len(chunks) >= 5  # 4 tokens + done
    assert any("[DONE]" in c for c in chunks)


@pytest.mark.asyncio
async def test_stream_processor_ndjson():
    async def token_source():
        yield "Hello"
        yield " world"

    processor = StreamProcessor(model="test", request_id="xyz")
    chunks = []
    async for chunk in processor.process_stream(token_source(), format="ndjson"):
        chunks.append(chunk)

    # Should have NDJSON lines
    assert all("\n" in c for c in chunks)
    import json

    first = json.loads(chunks[0])
    assert first["token"] == "Hello"


@pytest.mark.asyncio
async def test_stream_metrics():
    async def token_source():
        for i in range(10):
            yield f"token{i}"

    processor = StreamProcessor(model="test")
    async for _ in processor.process_stream(token_source()):
        pass

    assert processor.metrics.total_tokens == 10
    assert processor.metrics.first_token_ms > 0
    assert processor.metrics.total_duration_ms > 0
    assert processor.metrics.tokens_per_second > 0


def test_stream_buffer_word_boundary():
    buffer = StreamBuffer(flush_interval=10, min_buffer_size=3)

    # Short tokens should be buffered
    assert buffer.add("H") is None
    assert buffer.add("e") is None

    # Flush on word boundary (space)
    result = buffer.add("l ")
    assert result == "Hel "


def test_stream_buffer_flush():
    buffer = StreamBuffer()
    buffer.add("Hello")
    result = buffer.flush()
    assert result == "Hello"
    assert buffer.flush() == ""  # Empty after flush


@pytest.mark.asyncio
async def test_stream_multiplexer():
    mux = StreamMultiplexer()

    q1 = mux.subscribe()
    q2 = mux.subscribe()

    await mux.publish("hello")
    await mux.publish("world")

    assert await q1.get() == "hello"
    assert await q1.get() == "world"
    assert await q2.get() == "hello"
    assert await q2.get() == "world"


@pytest.mark.asyncio
async def test_stream_multiplexer_late_subscriber():
    mux = StreamMultiplexer()

    await mux.publish("first")

    # Late subscriber should get history
    q = mux.subscribe()
    assert await q.get() == "first"

    await mux.publish("second")
    assert await q.get() == "second"
