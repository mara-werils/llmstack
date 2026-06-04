"""Tests for widget routes."""

from __future__ import annotations
from fastapi.testclient import TestClient
from llmstack.gateway.main import create_app


class TestWidgetRoutes:
    def test_widget_config(self):
        app = create_app()
        c = TestClient(app)
        resp = c.get("/widget/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "chat_endpoint" in data

    def test_widget_embed(self):
        app = create_app()
        c = TestClient(app)
        resp = c.get("/widget/embed")
        assert resp.status_code == 200
        assert "LLMStack" in resp.text
