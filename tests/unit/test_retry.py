"""Tests for smart retry with provider fallback."""

import pytest

from llmstack.gateway.retry import (
    FallbackChain,
    RetryAttempt,
    RetryConfig,
    RetryResult,
    _compute_delay,
    retry_with_fallback,
)


class TestFallbackChain:
    def test_iterate_providers(self):
        chain = FallbackChain(providers=["a", "b", "c"])
        assert chain.next_provider() == "a"
        assert chain.next_provider() == "b"
        assert chain.next_provider() == "c"
        assert chain.next_provider() is None

    def test_exhausted(self):
        chain = FallbackChain(providers=["a"])
        assert chain.exhausted is False
        chain.next_provider()
        assert chain.exhausted is True

    def test_reset(self):
        chain = FallbackChain(providers=["a", "b"])
        chain.next_provider()
        chain.reset()
        assert chain.next_provider() == "a"


class TestComputeDelay:
    def test_exponential_growth(self):
        config = RetryConfig(initial_delay_ms=100, exponential_base=2.0, jitter=False)
        assert _compute_delay(0, config) == 100
        assert _compute_delay(1, config) == 200
        assert _compute_delay(2, config) == 400

    def test_max_delay_cap(self):
        config = RetryConfig(
            initial_delay_ms=1000,
            max_delay_ms=5000,
            exponential_base=10.0,
            jitter=False,
        )
        assert _compute_delay(5, config) == 5000

    def test_jitter_varies_delay(self):
        config = RetryConfig(initial_delay_ms=100, exponential_base=2.0, jitter=True)
        delays = {_compute_delay(1, config) for _ in range(20)}
        assert len(delays) > 1
        assert all(100 <= d <= 300 for d in delays)


class TestRetryAttemptAndResult:
    def test_retry_attempt_to_dict_rounds_latency(self):
        attempt = RetryAttempt(
            attempt=1, provider="openai", status_code=200, latency_ms=12.3456, success=True
        )
        d = attempt.to_dict()
        assert d["latency_ms"] == 12.3
        assert d["provider"] == "openai"

    def test_retry_attempt_timestamp_defaults_to_now(self):
        attempt = RetryAttempt(attempt=1, provider="local")
        assert attempt.timestamp > 0

    def test_retry_result_failed_attempts(self):
        result = RetryResult(
            attempts=[
                RetryAttempt(attempt=1, provider="a", success=False),
                RetryAttempt(attempt=2, provider="a", success=True),
            ]
        )
        assert result.failed_attempts == 1

    def test_retry_result_provider_switches(self):
        result = RetryResult(
            attempts=[
                RetryAttempt(attempt=1, provider="a"),
                RetryAttempt(attempt=2, provider="a"),
                RetryAttempt(attempt=3, provider="b"),
            ]
        )
        assert result.provider_switches == 1

    def test_retry_result_last_attempt(self):
        result = RetryResult()
        assert result.last_attempt is None
        attempt = RetryAttempt(attempt=1, provider="a")
        result.attempts.append(attempt)
        assert result.last_attempt is attempt

    def test_retry_result_to_dict(self):
        result = RetryResult(
            success=True,
            total_latency_ms=42.567,
            final_provider="openai",
            attempts=[RetryAttempt(attempt=1, provider="openai", success=True)],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["total_latency_ms"] == 42.6
        assert d["final_provider"] == "openai"
        assert len(d["attempts"]) == 1


class TestRetryWithFallback:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        async def handler(payload, provider=""):
            return {"result": "ok"}

        result = await retry_with_fallback(handler, {})
        assert result.success is True
        assert result.total_attempts == 1

    @pytest.mark.asyncio
    async def test_retry_on_exception(self):
        call_count = 0

        async def handler(payload, provider=""):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return {"result": "ok"}

        config = RetryConfig(max_retries=3, initial_delay_ms=1, jitter=False)
        result = await retry_with_fallback(handler, {}, config=config)
        assert result.success is True
        assert result.total_attempts == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        async def handler(payload, provider=""):
            raise ConnectionError("always fails")

        config = RetryConfig(max_retries=2, initial_delay_ms=1, jitter=False)
        result = await retry_with_fallback(handler, {}, config=config)
        assert result.success is False
        assert result.total_attempts == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_provider_fallback(self):
        providers_tried = []

        async def handler(payload, provider=""):
            providers_tried.append(provider)
            if provider == "primary":
                raise ConnectionError("primary down")
            return {"result": "ok"}

        config = RetryConfig(max_retries=3, initial_delay_ms=1, jitter=False)
        result = await retry_with_fallback(
            handler,
            {},
            config=config,
            fallback_providers=["primary", "backup"],
        )
        assert result.success is True
        assert "backup" in providers_tried

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        async def handler(payload, provider=""):
            raise ValueError("bad request")

        config = RetryConfig(max_retries=3, initial_delay_ms=1)
        result = await retry_with_fallback(handler, {}, config=config)
        assert result.success is False
        assert result.total_attempts == 1  # No retry for ValueError

    @pytest.mark.asyncio
    async def test_result_records_attempts(self):
        call_count = 0

        async def handler(payload, provider=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timeout")
            return {"result": "ok"}

        config = RetryConfig(max_retries=2, initial_delay_ms=1, jitter=False)
        result = await retry_with_fallback(handler, {}, config=config)
        assert len(result.attempts) == 2
        assert result.attempts[0].success is False
        assert result.attempts[1].success is True

    @pytest.mark.asyncio
    async def test_retryable_server_error_status_in_response_body(self):
        call_count = 0

        async def handler(payload, provider=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"error": {"type": "server_error", "status_code": 503}}
            return {"result": "ok"}

        config = RetryConfig(max_retries=2, initial_delay_ms=1, jitter=False)
        result = await retry_with_fallback(handler, {}, config=config)
        assert result.success is True
        assert result.attempts[0].status_code == 503
        assert result.attempts[0].success is False

    @pytest.mark.asyncio
    async def test_non_retryable_status_code_stops_immediately(self):
        async def handler(payload, provider=""):
            return {"error": {"type": "server_error", "status_code": 400}}

        config = RetryConfig(max_retries=3, initial_delay_ms=1, jitter=False)
        result = await retry_with_fallback(handler, {}, config=config)
        assert result.success is False
        assert result.total_attempts == 1
        assert result.attempts[0].status_code == 400
