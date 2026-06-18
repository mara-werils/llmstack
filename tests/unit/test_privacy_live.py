"""Tests for the live gateway privacy probe (powers `llmstack verify-private --live`)."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from llmstack.core.privacy import CRITICAL, INFO, WARNING
from llmstack.core.privacy_live import (
    _probe_live_auth,
    _probe_live_cors,
    _probe_live_providers,
    probe_live_gateway,
)

BASE_URL = "http://localhost:8000"


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient driven by per-path response maps."""

    def __init__(self, *, get_map=None, options_map=None, fail_all=False):
        self._get_map = get_map or {}
        self._options_map = options_map or {}
        self._fail_all = fail_all

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        if self._fail_all:
            raise httpx.ConnectError("refused")
        return self._get_map.get(url, _FakeResponse(404))

    async def options(self, url, **kwargs):
        if self._fail_all:
            raise httpx.ConnectError("refused")
        return self._options_map.get(url, _FakeResponse(200))


def _patch_client(client):
    return patch("llmstack.core.privacy_live.httpx.AsyncClient", return_value=client)


class TestProbeLiveGateway:
    @pytest.mark.asyncio
    async def test_unreachable_gateway_returns_info_only(self):
        client = _FakeAsyncClient(fail_all=True)
        with _patch_client(client):
            findings = await probe_live_gateway(BASE_URL)
        assert len(findings) == 1
        assert findings[0].severity == INFO
        assert findings[0].category == "live-probe"

    @pytest.mark.asyncio
    async def test_clean_gateway_reports_nothing(self):
        # A genuinely clean gateway enforces auth, so the unauthenticated
        # /v1/models probe gets a 401 (no model data, no auth warning).
        client = _FakeAsyncClient(
            get_map={
                f"{BASE_URL}/healthz": _FakeResponse(200),
                f"{BASE_URL}/v1/models": _FakeResponse(401),
            },
            options_map={
                f"{BASE_URL}/v1/models": _FakeResponse(200, headers={}),
            },
        )
        with _patch_client(client):
            findings = await probe_live_gateway(BASE_URL)
        assert findings == []

    @pytest.mark.asyncio
    async def test_detects_external_provider_open_cors_and_no_auth(self):
        client = _FakeAsyncClient(
            get_map={
                f"{BASE_URL}/healthz": _FakeResponse(200),
                f"{BASE_URL}/v1/models": _FakeResponse(
                    200,
                    json_data={
                        "object": "list",
                        "data": [
                            {"id": "gpt-4o", "owned_by": "openai"},
                            {"id": "llama3.2", "owned_by": "ollama"},
                        ],
                    },
                ),
            },
            options_map={
                f"{BASE_URL}/v1/models": _FakeResponse(
                    200, headers={"access-control-allow-origin": "*"}
                ),
            },
        )
        with _patch_client(client):
            findings = await probe_live_gateway(BASE_URL)

        categories = {f.category: f for f in findings}
        assert categories["live-providers"].severity == CRITICAL
        assert "openai" in categories["live-providers"].detail
        assert categories["live-cors"].severity == WARNING
        assert categories["live-auth"].severity == WARNING

    @pytest.mark.asyncio
    async def test_strips_trailing_slash_from_base_url(self):
        client = _FakeAsyncClient(fail_all=True)
        with _patch_client(client):
            findings = await probe_live_gateway(f"{BASE_URL}/")
        assert BASE_URL in findings[0].detail
        assert f"{BASE_URL}/" not in findings[0].detail


class TestProbeLiveProviders:
    @pytest.mark.asyncio
    async def test_no_external_providers(self):
        client = _FakeAsyncClient(
            get_map={
                f"{BASE_URL}/v1/models": _FakeResponse(
                    200, json_data={"data": [{"id": "llama3.2", "owned_by": "ollama"}]}
                )
            }
        )
        findings = await _probe_live_providers(client, BASE_URL)
        assert findings == []

    @pytest.mark.asyncio
    async def test_external_provider_via_x_llmstack(self):
        client = _FakeAsyncClient(
            get_map={
                f"{BASE_URL}/v1/models": _FakeResponse(
                    200,
                    json_data={
                        "data": [{"id": "claude-3", "x_llmstack": {"provider": "anthropic"}}]
                    },
                )
            }
        )
        findings = await _probe_live_providers(client, BASE_URL)
        assert len(findings) == 1
        assert "anthropic" in findings[0].detail

    @pytest.mark.asyncio
    async def test_models_endpoint_failure_returns_empty(self):
        client = _FakeAsyncClient(fail_all=True)
        findings = await _probe_live_providers(client, BASE_URL)
        assert findings == []

    @pytest.mark.asyncio
    async def test_models_endpoint_bad_status_returns_empty(self):
        client = _FakeAsyncClient(get_map={f"{BASE_URL}/v1/models": _FakeResponse(500)})
        findings = await _probe_live_providers(client, BASE_URL)
        assert findings == []


class TestProbeLiveCors:
    @pytest.mark.asyncio
    async def test_no_wildcard_cors_reports_nothing(self):
        client = _FakeAsyncClient(
            options_map={f"{BASE_URL}/v1/models": _FakeResponse(200, headers={})}
        )
        findings = await _probe_live_cors(client, BASE_URL)
        assert findings == []

    @pytest.mark.asyncio
    async def test_wildcard_cors_flagged(self):
        client = _FakeAsyncClient(
            options_map={
                f"{BASE_URL}/v1/models": _FakeResponse(
                    200, headers={"access-control-allow-origin": "*"}
                )
            }
        )
        findings = await _probe_live_cors(client, BASE_URL)
        assert len(findings) == 1
        assert findings[0].severity == WARNING

    @pytest.mark.asyncio
    async def test_request_failure_returns_empty(self):
        client = _FakeAsyncClient(fail_all=True)
        findings = await _probe_live_cors(client, BASE_URL)
        assert findings == []


class TestProbeLiveAuth:
    @pytest.mark.asyncio
    async def test_unauthenticated_200_flagged(self):
        client = _FakeAsyncClient(get_map={f"{BASE_URL}/v1/models": _FakeResponse(200)})
        findings = await _probe_live_auth(client, BASE_URL)
        assert len(findings) == 1
        assert findings[0].severity == WARNING

    @pytest.mark.asyncio
    async def test_auth_enforced_reports_nothing(self):
        client = _FakeAsyncClient(get_map={f"{BASE_URL}/v1/models": _FakeResponse(401)})
        findings = await _probe_live_auth(client, BASE_URL)
        assert findings == []

    @pytest.mark.asyncio
    async def test_request_failure_returns_empty(self):
        client = _FakeAsyncClient(fail_all=True)
        findings = await _probe_live_auth(client, BASE_URL)
        assert findings == []
