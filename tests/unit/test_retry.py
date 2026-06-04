"""Tests for smart retry with provider fallback."""

import pytest

from llmstack.gateway.retry import (
    RetryConfig,
    FallbackChain,
    retry_with_fallback,
    _compute_delay,
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
