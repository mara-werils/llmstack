"""Tests for the /v1/onboarding readiness route."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.routes import onboarding as onboarding_route


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(onboarding_route.router, prefix="/v1")
    return TestClient(app)


def test_onboarding_reports_not_ready_when_ollama_unreachable(client):
    # 127.0.0.1:1 is loopback-only and refuses -> Ollama not running.
    body = client.get("/v1/onboarding", params={"ollama_url": "http://127.0.0.1:1"}).json()
    assert body["ready"] is False
    assert body["ollama"]["running"] is False
    assert body["ollama"]["url"] == "http://127.0.0.1:1"
    assert body["hints"]


def test_onboarding_exposes_recommended_models_and_hardware(client):
    body = client.get("/v1/onboarding", params={"ollama_url": "http://127.0.0.1:1"}).json()
    assert body["recommended"]["chat_model"]["name"]
    assert body["recommended"]["embed_model"]["name"]
    assert body["hardware"]["cpu_cores"] >= 1
    assert "gpu_vendor" in body["hardware"]


def test_onboarding_ready_when_models_present(client, monkeypatch):
    from llmstack.core.onboarding import OllamaStatus

    # Force a "running with both recommended models" status.
    def fake_probe(url, **kwargs):
        from llmstack.core.hardware import detect_hardware
        from llmstack.core.onboarding import recommend_embed_model, recommend_model

        hw = detect_hardware()
        return OllamaStatus(
            running=True,
            models=(recommend_model(hw).name, recommend_embed_model(hw).name),
        )

    monkeypatch.setattr(onboarding_route, "probe_ollama", fake_probe)
    body = client.get("/v1/onboarding").json()
    assert body["ready"] is True
    assert body["chat_model"]["ready"] is True
    assert body["embed_model"]["ready"] is True
