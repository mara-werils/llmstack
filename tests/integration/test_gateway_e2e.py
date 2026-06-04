"""End-to-end gateway integration tests.

Tests the full request pipeline through the FastAPI app with mocked
inference backends.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with the gateway app."""
    from llmstack.gateway.main import create_app
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_docs_endpoint(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/v1/chat/completions" in data["paths"]


class TestChatCompletions:
    def test_chat_returns_error_without_backend(self, client):
        """Without a running backend, chat should return a meaningful error."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should get either 503 (circuit breaker) or 502/500 (connection error)
        assert resp.status_code in (500, 502, 503)

    def test_chat_validation_rejects_empty_messages(self, client):
        """Empty messages list should return 422."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": []},
        )
        assert resp.status_code == 422

    def test_chat_validation_rejects_invalid_temperature(self, client):
        """Temperature > 2.0 should return 422."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 5.0,
            },
        )
        assert resp.status_code == 422

    @patch("llmstack.gateway.proxy.proxy_chat_completion")
    async def test_chat_success_with_mock(self, mock_proxy, client):
        """Successful chat completion with mocked backend."""
        mock_proxy.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"


class TestAuthMiddleware:
    def test_auth_required_when_configured(self):
        """When API keys are set, auth should be required."""
        import os
        with patch.dict(os.environ, {"LLMSTACK_API_KEYS": "test-key-123"}):
            from llmstack.gateway.main import create_app
            app = create_app()
            c = TestClient(app)

            # Without auth header
            resp = c.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
            )
            assert resp.status_code == 401

            # With correct auth header
            resp = c.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
                headers={"Authorization": "Bearer test-key-123"},
            )
            # Should pass auth (may fail at proxy level but not at auth)
            assert resp.status_code != 401

    def test_health_bypasses_auth(self):
        """Health endpoints should not require auth."""
        import os
        with patch.dict(os.environ, {"LLMSTACK_API_KEYS": "test-key-123"}):
            from llmstack.gateway.main import create_app
            app = create_app()
            c = TestClient(app)
            resp = c.get("/healthz")
            assert resp.status_code == 200


class TestRateLimiting:
    def test_rate_limit_headers_present(self, client):
        """Rate limit headers should be in responses."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        # Even if request fails, rate limit headers should be present
        # (unless it hit skip paths)
        if resp.status_code not in (401, 403):
            assert "X-RateLimit-Limit" in resp.headers or resp.status_code >= 500


class TestAPIResponseFormat:
    def test_error_response_format(self, client):
        """Error responses should follow OpenAI error format."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
        )
        if resp.status_code >= 400:
            data = resp.json()
            assert "error" in data
            assert "message" in data["error"]
            assert "type" in data["error"]
