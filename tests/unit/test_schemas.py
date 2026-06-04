"""Tests for gateway request schemas."""

from __future__ import annotations
import pytest
from pydantic import ValidationError
from llmstack.gateway.schemas import ChatCompletionRequest, ChatMessage


class TestChatMessage:
    def test_valid_message(self):
        m = ChatMessage(role="user", content="Hello")
        assert m.role == "user"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hi")


class TestChatCompletionRequest:
    def test_valid_request(self):
        r = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert r.model == "test"
        assert r.stream is False

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="test", messages=[])

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=5.0,
            )

    def test_defaults(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
        )
        assert r.temperature == 1.0
        assert r.top_p == 1.0
        assert r.max_tokens is None
