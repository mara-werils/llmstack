"""Tests for llmstack.core.health.wait_healthy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmstack.core.health import wait_healthy


class _FakeAsyncClient:
    def __init__(self, get_side_effect):
        self._get_side_effect = get_side_effect
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        self.calls += 1
        result = self._get_side_effect[self.calls - 1]
        if isinstance(result, Exception):
            raise result
        return result


def _resp(status_code):
    return MagicMock(status_code=status_code)


@pytest.mark.asyncio
async def test_wait_healthy_succeeds_on_first_attempt():
    client = _FakeAsyncClient([_resp(200)])
    with (
        patch("llmstack.core.health.httpx.AsyncClient", return_value=client),
        patch("llmstack.core.health.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        result = await wait_healthy("http://svc/health", timeout=10, interval=1.0)
    assert result is True
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_wait_healthy_retries_then_succeeds():
    client = _FakeAsyncClient([_resp(503), _resp(200)])
    with (
        patch("llmstack.core.health.httpx.AsyncClient", return_value=client),
        patch("llmstack.core.health.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        result = await wait_healthy("http://svc/health", timeout=10, interval=1.0)
    assert result is True
    assert mock_sleep.call_count == 1


@pytest.mark.asyncio
async def test_wait_healthy_swallows_http_errors_and_keeps_polling():
    client = _FakeAsyncClient([httpx.ConnectError("down"), _resp(200)])
    with (
        patch("llmstack.core.health.httpx.AsyncClient", return_value=client),
        patch("llmstack.core.health.asyncio.sleep", new=AsyncMock()),
    ):
        result = await wait_healthy("http://svc/health", timeout=10, interval=1.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_healthy_times_out_returns_false():
    client = _FakeAsyncClient([_resp(503)] * 10)
    with (
        patch("llmstack.core.health.httpx.AsyncClient", return_value=client),
        patch("llmstack.core.health.asyncio.sleep", new=AsyncMock()),
    ):
        result = await wait_healthy("http://svc/health", timeout=3, interval=1.0)
    assert result is False
