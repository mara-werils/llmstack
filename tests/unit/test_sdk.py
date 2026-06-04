"""Tests for SDK client convenience methods and types."""

from __future__ import annotations


from llmstack.sdk.client import LLMStackError, _parse_sse_line
from llmstack.sdk.types import ChatResponse


class TestParseSSE:
    def test_data_line(self):
        result = _parse_sse_line('data: {"key": "value"}')
        assert result == {"key": "value"}

    def test_done_line(self):
        assert _parse_sse_line("data: [DONE]") is None

    def test_empty_line(self):
        assert _parse_sse_line("") is None

    def test_comment_line(self):
        assert _parse_sse_line(": keep-alive") is None

    def test_invalid_json(self):
        assert _parse_sse_line("data: {invalid}") is None


class TestChatResponse:
    def test_from_dict(self):
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = ChatResponse.from_dict(data)
        assert resp.id == "chatcmpl-123"
        assert resp.model == "llama3.2"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage.total_tokens == 15

    def test_from_dict_with_cache_headers(self):
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "llama3.2",
            "choices": [],
        }
        headers = {"x-cache": "HIT", "x-cache-age": "42"}
        resp = ChatResponse.from_dict(data, headers=headers)
        assert resp.cached is True
        assert resp.cache_age == 42


class TestLLMStackError:
    def test_error_message(self):
        err = LLMStackError(404, {"error": "not found"})
        assert err.status_code == 404
        assert "404" in str(err)
