"""Tests for ask Ollama provisioning helpers (first-run UX)."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llmstack.ask.ollama import (
    OllamaStatus,
    _model_exists,
    _pull_with_progress,
    check_ollama,
    ensure_models,
    install_hint,
)


class _FakeResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeClient:
    """Minimal AsyncClient stand-in: /api/show probe + a pull guard."""

    def __init__(self, exists: bool = True) -> None:
        self._exists = exists
        self.pull_called = False

    async def post(self, url: str, json: dict | None = None) -> _FakeResp:
        return _FakeResp(200 if self._exists else 404)

    def stream(self, *args, **kwargs):  # pragma: no cover - guard
        self.pull_called = True
        raise AssertionError("pull must not run when the model already exists")


def test_install_hint_not_installed_mentions_installation() -> None:
    msg = install_hint(OllamaStatus(reachable=False, installed=False), "http://localhost:11434")
    assert "not installed" in msg.lower()
    # Platform-aware install command should be present.
    assert "ollama.com" in msg or "brew install ollama" in msg


def test_install_hint_installed_but_not_running() -> None:
    msg = install_hint(OllamaStatus(reachable=False, installed=True), "http://localhost:11434")
    assert "ollama serve" in msg
    assert "not running" in msg.lower()


@pytest.mark.asyncio
async def test_ensure_models_skips_present_and_dedupes() -> None:
    client = _FakeClient(exists=True)
    # Duplicates and empty strings must be collapsed; nothing should be pulled.
    await ensure_models(
        "http://localhost:11434", ["llama3.2", "llama3.2", "", "nomic-embed-text"], client=client
    )
    assert client.pull_called is False


def test_install_hint_linux():
    with patch.object(sys, "platform", "linux"):
        msg = install_hint(OllamaStatus(reachable=False, installed=False), "http://localhost:11434")
    assert "install.sh" in msg


def test_install_hint_other_platform():
    with patch.object(sys, "platform", "win32"):
        msg = install_hint(OllamaStatus(reachable=False, installed=False), "http://localhost:11434")
    assert "Download the installer" in msg


class _FakeGetResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeGetClient:
    def __init__(self, resp=None, error=None):
        self._resp = resp
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        if self._error:
            raise self._error
        return self._resp


@pytest.mark.asyncio
async def test_check_ollama_reachable_returns_version():
    fake_client = _FakeGetClient(resp=_FakeGetResp(200, {"version": "0.5.1"}))
    with patch("llmstack.ask.ollama.httpx.AsyncClient", return_value=fake_client):
        status = await check_ollama("http://localhost:11434/")
    assert status.reachable is True
    assert status.installed is True
    assert status.version == "0.5.1"


@pytest.mark.asyncio
async def test_check_ollama_unreachable_but_binary_installed():
    fake_client = _FakeGetClient(resp=_FakeGetResp(500))
    with (
        patch("llmstack.ask.ollama.httpx.AsyncClient", return_value=fake_client),
        patch("llmstack.ask.ollama.shutil.which", return_value="/usr/bin/ollama"),
    ):
        status = await check_ollama("http://localhost:11434")
    assert status.reachable is False
    assert status.installed is True


@pytest.mark.asyncio
async def test_check_ollama_http_error_not_installed():
    fake_client = _FakeGetClient(error=httpx.ConnectError("refused"))
    with (
        patch("llmstack.ask.ollama.httpx.AsyncClient", return_value=fake_client),
        patch("llmstack.ask.ollama.shutil.which", return_value=None),
    ):
        status = await check_ollama("http://localhost:11434")
    assert status.reachable is False
    assert status.installed is False


@pytest.mark.asyncio
async def test_model_exists_http_error_returns_false():
    class RaisingClient:
        async def post(self, url, json=None):
            raise httpx.ConnectError("refused")

    exists = await _model_exists(RaisingClient(), "http://localhost:11434", "llama3.2")
    assert exists is False


class _FakeStreamCtx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, json=None, timeout=None):
        return _FakeStreamCtx(self._lines)


@pytest.mark.asyncio
async def test_pull_with_progress_renders_layers_and_skips_bad_lines():
    lines = [
        "",
        '{"digest":"sha256:abcdef123456","total":100,"completed":50}',
        "not-json{",
        '{"digest":"sha256:abcdef123456","total":100,"completed":100}',
        '{"digest":"sha256:other000000","total":10,"completed":10}',
    ]
    client = _FakeStreamClient(lines)
    await _pull_with_progress(client, "http://localhost:11434", "llama3.2")


@pytest.mark.asyncio
async def test_pull_with_progress_raises_on_error_payload():
    client = _FakeStreamClient(['{"error":"model not found"}'])
    with pytest.raises(RuntimeError, match="model not found"):
        await _pull_with_progress(client, "http://localhost:11434", "llama3.2")


@pytest.mark.asyncio
async def test_ensure_models_pulls_missing_and_closes_owned_client():
    with (
        patch("llmstack.ask.ollama._model_exists", new=AsyncMock(return_value=False)),
        patch("llmstack.ask.ollama._pull_with_progress", new=AsyncMock()) as mock_pull,
    ):
        await ensure_models("http://localhost:11434", ["llama3.2"])
    mock_pull.assert_awaited_once()
