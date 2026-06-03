"""Comprehensive tests for ConversationEngine — multi-turn chat with context."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmstack.ask.conversation import ConversationEngine, ConversationTurn
from llmstack.ask.parsers import TextChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ConversationEngine:
    return ConversationEngine(
        ollama_url="http://test:11434",
        model="test-model",
    )


@pytest.fixture
def context_chunks() -> list[tuple[TextChunk, float]]:
    return [
        (TextChunk(content="def foo(): pass", source="foo.py", start_line=1, end_line=1), 0.9),
        (TextChunk(content="class Bar: ...", source="bar.py", start_line=5, end_line=10), 0.7),
    ]


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------


class TestConversationTurn:
    def test_fields(self) -> None:
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.sources == []

    def test_with_sources(self) -> None:
        turn = ConversationTurn(role="assistant", content="answer", sources=["a.py:1-5"])
        assert turn.sources == ["a.py:1-5"]


# ---------------------------------------------------------------------------
# ConversationEngine init
# ---------------------------------------------------------------------------


class TestConversationEngineInit:
    def test_default_config(self) -> None:
        e = ConversationEngine()
        assert e.ollama_url == "http://localhost:11434"
        assert e.model == "llama3.2"
        assert e.history == []
        assert e.turn_count == 0

    def test_custom_url_strips_slash(self) -> None:
        e = ConversationEngine(ollama_url="http://host:1234/")
        assert e.ollama_url == "http://host:1234"

    def test_custom_system_prompt(self) -> None:
        e = ConversationEngine(system_prompt="Custom prompt")
        assert e._system_prompt == "Custom prompt"

    def test_default_system_prompt_used(self) -> None:
        e = ConversationEngine()
        assert "knowledgeable assistant" in e._system_prompt

    def test_git_context_stored(self) -> None:
        e = ConversationEngine(git_context="branch: main")
        assert e._git_context == "branch: main"


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


class TestHistoryManagement:
    def test_history_is_copy(self, engine: ConversationEngine) -> None:
        engine._history.append(ConversationTurn(role="user", content="hi"))
        h = engine.history
        h.append(ConversationTurn(role="user", content="extra"))
        assert len(engine._history) == 1  # original not modified

    def test_turn_count_counts_user_only(self, engine: ConversationEngine) -> None:
        engine._history.append(ConversationTurn(role="user", content="q1"))
        engine._history.append(ConversationTurn(role="assistant", content="a1"))
        engine._history.append(ConversationTurn(role="user", content="q2"))
        assert engine.turn_count == 2

    def test_clear(self, engine: ConversationEngine) -> None:
        engine._history.append(ConversationTurn(role="user", content="hi"))
        engine.clear()
        assert engine.history == []
        assert engine.turn_count == 0


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_system_prompt_first(self, engine: ConversationEngine) -> None:
        msgs = engine._build_messages("hello")
        assert msgs[0]["role"] == "system"

    def test_current_message_last(self, engine: ConversationEngine) -> None:
        msgs = engine._build_messages("hello")
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "hello"

    def test_includes_history(self, engine: ConversationEngine) -> None:
        engine._history.append(ConversationTurn(role="user", content="q1"))
        engine._history.append(ConversationTurn(role="assistant", content="a1"))
        msgs = engine._build_messages("q2")
        # system + history(2) + current
        assert len(msgs) == 4
        assert msgs[1]["content"] == "q1"
        assert msgs[2]["content"] == "a1"

    def test_git_context_appended_to_system(self) -> None:
        e = ConversationEngine(git_context="branch: main")
        msgs = e._build_messages("hi")
        assert "branch: main" in msgs[0]["content"]

    def test_history_truncated_to_20_entries(self, engine: ConversationEngine) -> None:
        for i in range(30):
            engine._history.append(ConversationTurn(role="user", content=f"q{i}"))
            engine._history.append(ConversationTurn(role="assistant", content=f"a{i}"))
        msgs = engine._build_messages("final")
        # system + 20 history entries + current = 22
        assert len(msgs) == 22


# ---------------------------------------------------------------------------
# ConversationEngine.ask
# ---------------------------------------------------------------------------


class TestConversationAsk:
    @pytest.mark.asyncio
    async def test_ask_streams_tokens(
        self,
        engine: ConversationEngine,
        context_chunks: list[tuple[TextChunk, float]],
    ) -> None:
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " there"}, "done": True}),
        ]

        mock_resp = _make_stream_mock(lines)

        with patch.object(engine._client, "stream", return_value=mock_resp):
            tokens = []
            async for token in engine.ask("what is foo?", context_chunks):
                tokens.append(token)
            assert tokens == ["Hello", " there"]

    @pytest.mark.asyncio
    async def test_ask_records_history(
        self,
        engine: ConversationEngine,
        context_chunks: list[tuple[TextChunk, float]],
    ) -> None:
        lines = [json.dumps({"message": {"content": "answer"}, "done": True})]

        mock_resp = _make_stream_mock(lines)

        with patch.object(engine._client, "stream", return_value=mock_resp):
            async for _ in engine.ask("question?", context_chunks):
                pass

        assert engine.turn_count == 1
        assert engine._history[0].role == "user"
        assert engine._history[0].content == "question?"
        assert engine._history[1].role == "assistant"
        assert engine._history[1].content == "answer"
        assert len(engine._history[1].sources) == 2

    @pytest.mark.asyncio
    async def test_ask_empty_context(self, engine: ConversationEngine) -> None:
        lines = [json.dumps({"message": {"content": "ok"}, "done": True})]

        mock_resp = _make_stream_mock(lines)

        with patch.object(engine._client, "stream", return_value=mock_resp):
            tokens = []
            async for token in engine.ask("question?", []):
                tokens.append(token)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_ask_handles_empty_lines(
        self,
        engine: ConversationEngine,
        context_chunks: list[tuple[TextChunk, float]],
    ) -> None:
        lines = [
            "",
            "  ",
            json.dumps({"message": {"content": "ok"}, "done": True}),
        ]

        mock_resp = _make_stream_mock(lines)

        with patch.object(engine._client, "stream", return_value=mock_resp):
            tokens = []
            async for token in engine.ask("q?", context_chunks):
                tokens.append(token)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_ask_handles_json_decode_error(
        self,
        engine: ConversationEngine,
        context_chunks: list[tuple[TextChunk, float]],
    ) -> None:
        lines = [
            "not json",
            json.dumps({"message": {"content": "ok"}, "done": True}),
        ]

        mock_resp = _make_stream_mock(lines)

        with patch.object(engine._client, "stream", return_value=mock_resp):
            tokens = []
            async for token in engine.ask("q?", context_chunks):
                tokens.append(token)
            assert tokens == ["ok"]


# ---------------------------------------------------------------------------
# ConversationEngine.close
# ---------------------------------------------------------------------------


class TestConversationClose:
    @pytest.mark.asyncio
    async def test_close(self, engine: ConversationEngine) -> None:
        with patch.object(engine._client, "aclose", new_callable=AsyncMock) as mock_close:
            await engine.close()
            mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_iter(items):
    for item in items:
        yield item


def _make_stream_mock(lines: list[str]):
    """Create a mock that works as an async context manager with aiter_lines."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = lambda: _async_iter(lines)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp
