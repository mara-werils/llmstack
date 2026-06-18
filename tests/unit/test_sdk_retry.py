"""Tests for SDK retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmstack.sdk.retry import (
    RetryConfig,
    _calculate_delay,
    _get_retry_after,
    async_retry,
    sync_retry,
)


class TestRetryConfig:
    def test_defaults(self):
        c = RetryConfig()
        assert c.max_retries == 3
        assert 429 in c.retry_on_status
        assert c.backoff_factor == 2.0


class TestCalculateDelay:
    def test_first_attempt(self):
        c = RetryConfig(jitter=False)
        delay = _calculate_delay(0, c)
        assert delay == 0.5

    def test_exponential_backoff(self):
        c = RetryConfig(jitter=False)
        d0 = _calculate_delay(0, c)
        d1 = _calculate_delay(1, c)
        d2 = _calculate_delay(2, c)
        assert d1 == d0 * 2
        assert d2 == d1 * 2

    def test_max_delay_cap(self):
        c = RetryConfig(jitter=False, max_delay=5.0)
        delay = _calculate_delay(10, c)
        assert delay <= 5.0

    def test_retry_after_override(self):
        c = RetryConfig(jitter=False)
        delay = _calculate_delay(0, c, retry_after=10.0)
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        c = RetryConfig(jitter=True)
        delays = {_calculate_delay(1, c) for _ in range(20)}
        assert len(delays) > 1  # should not all be the same


def _fake_response(status_code, headers=None):
    return MagicMock(status_code=status_code, headers=headers or {})


class TestGetRetryAfter:
    def test_missing_header_returns_none(self):
        assert _get_retry_after(_fake_response(429, {})) is None

    def test_invalid_value_returns_none(self):
        assert _get_retry_after(_fake_response(429, {"Retry-After": "not-a-number"})) is None

    def test_valid_value_returned(self):
        assert _get_retry_after(_fake_response(429, {"Retry-After": "2.5"})) == 2.5


class TestSyncRetry:
    def test_succeeds_on_first_try_without_sleeping(self):
        func = MagicMock(return_value=_fake_response(200))
        config = RetryConfig(max_retries=3)
        with patch("llmstack.sdk.retry.time.sleep") as mock_sleep:
            response = sync_retry(func, config)
        assert response.status_code == 200
        mock_sleep.assert_not_called()

    def test_retries_on_retryable_status_then_succeeds(self):
        func = MagicMock(
            side_effect=[_fake_response(503), _fake_response(503), _fake_response(200)]
        )
        config = RetryConfig(max_retries=3, jitter=False)
        with patch("llmstack.sdk.retry.time.sleep") as mock_sleep:
            response = sync_retry(func, config)
        assert response.status_code == 200
        assert mock_sleep.call_count == 2

    def test_returns_last_response_when_retries_exhausted(self):
        func = MagicMock(return_value=_fake_response(503))
        config = RetryConfig(max_retries=1, jitter=False)
        with patch("llmstack.sdk.retry.time.sleep"):
            response = sync_retry(func, config)
        assert response.status_code == 503
        assert func.call_count == 2

    def test_retries_on_connection_error_then_succeeds(self):
        func = MagicMock(side_effect=[httpx.ConnectError("down"), _fake_response(200)])
        config = RetryConfig(max_retries=2, jitter=False)
        with patch("llmstack.sdk.retry.time.sleep") as mock_sleep:
            response = sync_retry(func, config)
        assert response.status_code == 200
        mock_sleep.assert_called_once()

    def test_raises_after_exhausting_retries_on_connection_error(self):
        func = MagicMock(side_effect=httpx.ReadTimeout("timeout"))
        config = RetryConfig(max_retries=1, jitter=False)
        with patch("llmstack.sdk.retry.time.sleep"):
            with pytest.raises(httpx.ReadTimeout):
                sync_retry(func, config)

    def test_no_attempts_raises_runtime_error(self):
        config = RetryConfig(max_retries=-1)
        with pytest.raises(RuntimeError, match="Retry exhausted"):
            sync_retry(MagicMock(), config)


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try_without_sleeping(self):
        func = AsyncMock(return_value=_fake_response(200))
        config = RetryConfig(max_retries=3)
        with patch("llmstack.sdk.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            response = await async_retry(func, config)
        assert response.status_code == 200
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_retryable_status_then_succeeds(self):
        func = AsyncMock(
            side_effect=[_fake_response(429), _fake_response(429), _fake_response(200)]
        )
        config = RetryConfig(max_retries=3, jitter=False)
        with patch("llmstack.sdk.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            response = await async_retry(func, config)
        assert response.status_code == 200
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_last_response_when_retries_exhausted(self):
        func = AsyncMock(return_value=_fake_response(503))
        config = RetryConfig(max_retries=1, jitter=False)
        with patch("llmstack.sdk.retry.asyncio.sleep", new=AsyncMock()):
            response = await async_retry(func, config)
        assert response.status_code == 503
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_connection_error_then_succeeds(self):
        func = AsyncMock(side_effect=[httpx.ConnectTimeout("down"), _fake_response(200)])
        config = RetryConfig(max_retries=2, jitter=False)
        with patch("llmstack.sdk.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            response = await async_retry(func, config)
        assert response.status_code == 200
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries_on_connection_error(self):
        func = AsyncMock(side_effect=httpx.ConnectError("down"))
        config = RetryConfig(max_retries=1, jitter=False)
        with patch("llmstack.sdk.retry.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(httpx.ConnectError):
                await async_retry(func, config)

    @pytest.mark.asyncio
    async def test_no_attempts_raises_runtime_error(self):
        config = RetryConfig(max_retries=-1)
        with pytest.raises(RuntimeError, match="Retry exhausted"):
            await async_retry(AsyncMock(), config)
