"""Tests for llmstack.gateway.main: app factory, lifespan, and init helpers."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmstack.gateway.main import (
    _init_observe,
    _init_providers,
    _init_router,
    create_app,
    lifespan,
)


def test_serve_ui_root_returns_index_html():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_cors_warning_logged_in_production(monkeypatch, caplog):
    monkeypatch.setenv("LLMSTACK_CORS_ORIGINS", "*")
    monkeypatch.setenv("LLMSTACK_ENV", "production")
    monkeypatch.delenv("LLMSTACK_API_KEYS", raising=False)
    with caplog.at_level(logging.WARNING, logger="llmstack.gateway.main"):
        create_app()
    assert any("CORS is set to allow all origins" in r.message for r in caplog.records)


def test_no_cors_warning_outside_production(monkeypatch, caplog):
    monkeypatch.setenv("LLMSTACK_CORS_ORIGINS", "*")
    monkeypatch.delenv("LLMSTACK_ENV", raising=False)
    with caplog.at_level(logging.WARNING, logger="llmstack.gateway.main"):
        create_app()
    assert not any("CORS is set to allow all origins" in r.message for r in caplog.records)


def test_auth_middleware_added_when_api_keys_set(monkeypatch):
    monkeypatch.setenv("LLMSTACK_API_KEYS", "sekret123")
    app = create_app()
    client = TestClient(app)

    unauthed = client.get("/widget/config")
    assert unauthed.status_code == 401

    authed = client.get("/widget/config", headers={"Authorization": "Bearer sekret123"})
    assert authed.status_code == 200


def test_no_auth_middleware_when_api_keys_unset(monkeypatch):
    monkeypatch.delenv("LLMSTACK_API_KEYS", raising=False)
    app = create_app()
    client = TestClient(app)
    resp = client.get("/widget/config")
    assert resp.status_code == 200


# --- _init_router -----------------------------------------------------------


def test_init_router_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("LLMSTACK_ROUTER_CONFIG", raising=False)
    _init_router()  # should not raise


def test_init_router_invalid_json_warns(monkeypatch, caplog):
    monkeypatch.setenv("LLMSTACK_ROUTER_CONFIG", "{not json")
    with caplog.at_level(logging.WARNING, logger="llmstack.gateway.main"):
        _init_router()
    assert any("not valid JSON" in r.message for r in caplog.records)


def test_init_router_disabled_flag_noop(monkeypatch):
    monkeypatch.setenv("LLMSTACK_ROUTER_CONFIG", json.dumps({"enabled": False}))
    _init_router()  # should not raise


def test_init_router_no_models_warns(monkeypatch, caplog):
    monkeypatch.setenv("LLMSTACK_ROUTER_CONFIG", json.dumps({"enabled": True, "models": []}))
    with caplog.at_level(logging.WARNING, logger="llmstack.gateway.main"):
        _init_router()
    assert any("no models configured" in r.message for r in caplog.records)


def test_init_router_full_success(monkeypatch):
    cfg = {
        "enabled": True,
        "strategy": "cost",
        "models": [
            {"name": "llama3.2:1b", "tier": "simple", "quality_score": 0.6},
            {"name": "llama3.1:70b", "tier": "complex", "quality_score": 1.0},
        ],
    }
    monkeypatch.setenv("LLMSTACK_ROUTER_CONFIG", json.dumps(cfg))

    with patch("llmstack.gateway.router._state.init_router") as mock_init:
        _init_router()

    mock_init.assert_called_once()
    router_arg, stats_arg = mock_init.call_args[0]
    assert len(router_arg.models) == 2
    assert stats_arg._largest_model == "llama3.1:70b"


# --- _init_providers ----------------------------------------------------------


def test_init_providers_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("LLMSTACK_PROVIDERS_CONFIG", raising=False)
    _init_providers()  # should not raise


def test_init_providers_invalid_json_warns(monkeypatch, caplog):
    monkeypatch.setenv("LLMSTACK_PROVIDERS_CONFIG", "{not json")
    with caplog.at_level(logging.WARNING, logger="llmstack.gateway.main"):
        _init_providers()
    assert any("not valid JSON" in r.message for r in caplog.records)


def test_init_providers_disabled_flag_noop(monkeypatch):
    monkeypatch.setenv("LLMSTACK_PROVIDERS_CONFIG", json.dumps({"enabled": False}))
    _init_providers()  # should not raise


def test_init_providers_full_success(monkeypatch):
    cfg = {
        "enabled": True,
        "providers": [
            {
                "name": "openai",
                "api_key_env": "MY_OPENAI_KEY",
                "models": [{"name": "gpt-4o"}],
                "fallback": ["local"],
            },
            {"name": "unknown-vendor"},
            {"name": "anthropic", "enabled": False},
        ],
    }
    monkeypatch.setenv("LLMSTACK_PROVIDERS_CONFIG", json.dumps(cfg))
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-test-123")

    with patch("llmstack.gateway.providers.registry.init_registry") as mock_init:
        _init_providers()

    mock_init.assert_called_once()
    registry_arg = mock_init.call_args[0][0]
    assert registry_arg.has_provider("local")
    assert registry_arg.has_provider("openai")
    assert not registry_arg.has_provider("anthropic")  # disabled, skipped


# --- _init_observe -------------------------------------------------------------


def test_init_observe_sets_global_state():
    with patch("llmstack.observe._state.init_observe") as mock_init:
        _init_observe()
    mock_init.assert_called_once()
    kwargs = mock_init.call_args.kwargs
    assert "trace_store" in kwargs
    assert "scorer" in kwargs
    assert "tracker" in kwargs
    assert "ab_manager" in kwargs


# --- lifespan -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_full_cycle_drains_in_flight_requests():
    fake_cache = MagicMock()
    fake_cache.close = AsyncMock()
    app = FastAPI()

    with (
        patch("llmstack.gateway.cache.get_cache", new=AsyncMock(return_value=fake_cache)),
        patch("llmstack.gateway.main._init_router") as mock_router,
        patch("llmstack.gateway.main._init_providers") as mock_providers,
        patch("llmstack.gateway.main._init_observe") as mock_observe,
        patch(
            "llmstack.gateway.middleware.metrics.get_active_requests",
            side_effect=[1, 0],
        ),
        patch("llmstack.gateway.main.asyncio.sleep", new=AsyncMock()) as mock_sleep,
        patch("llmstack.gateway.proxy.close_pool", new=AsyncMock()) as mock_close_pool,
    ):
        async with lifespan(app):
            pass

    mock_router.assert_called_once()
    mock_providers.assert_called_once()
    mock_observe.assert_called_once()
    mock_sleep.assert_awaited_once()
    mock_close_pool.assert_awaited_once()
    fake_cache.close.assert_awaited_once()
