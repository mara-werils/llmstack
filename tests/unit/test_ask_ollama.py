"""Tests for ask Ollama provisioning helpers (first-run UX)."""

from __future__ import annotations

import pytest

from llmstack.ask.ollama import OllamaStatus, ensure_models, install_hint


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
    await ensure_models("http://localhost:11434", ["llama3.2", "llama3.2", "", "nomic-embed-text"], client=client)
    assert client.pull_called is False
