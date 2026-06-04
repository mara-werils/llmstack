"""Comprehensive tests for the AskEngine — load, ask, ask_full, streaming."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmstack.ask.engine import (
    AskEngine,
    AskResult,
    SourceRef,
    _build_context,
    _build_sources,
)
from llmstack.ask.parsers import TextChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(content="def hello(): pass", source="a.py", start_line=1, end_line=3),
        TextChunk(content="class Foo: ...", source="b.py", start_line=10, end_line=20),
        TextChunk(content="import os\nimport sys", source="c.py", start_line=1, end_line=2),
    ]


@pytest.fixture
def scored_chunks(sample_chunks: list[TextChunk]) -> list[tuple[TextChunk, float]]:
    return [(sample_chunks[0], 0.95), (sample_chunks[1], 0.80), (sample_chunks[2], 0.60)]


@pytest.fixture
def engine() -> AskEngine:
    return AskEngine(ollama_url="http://test:11434", model="test-model")


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_single_chunk(self) -> None:
        chunk = TextChunk(content="hello", source="f.py", start_line=1, end_line=5)
        ctx = _build_context([(chunk, 0.9)])
        assert "f.py:1-5" in ctx
        assert "hello" in ctx

    def test_multiple_chunks(self, scored_chunks: list[tuple[TextChunk, float]]) -> None:
        ctx = _build_context(scored_chunks)
        assert "a.py:1-3" in ctx
        assert "b.py:10-20" in ctx
        assert "c.py:1-2" in ctx

    def test_empty_list(self) -> None:
        ctx = _build_context([])
        assert ctx == ""

    def test_context_contains_separators(
        self, scored_chunks: list[tuple[TextChunk, float]]
    ) -> None:
        ctx = _build_context(scored_chunks)
        assert ctx.count("---") >= 4  # at least 2 per chunk (before/after)


# ---------------------------------------------------------------------------
# _build_sources
# ---------------------------------------------------------------------------


class TestBuildSources:
    def test_single_source(self) -> None:
        chunk = TextChunk(content="short", source="x.py", start_line=5, end_line=10)
        sources = _build_sources([(chunk, 0.8765)])
        assert len(sources) == 1
        assert sources[0].file == "x.py"
        assert sources[0].lines == "5-10"
        assert sources[0].relevance == 0.8765

    def test_preview_truncated_to_100_chars(self) -> None:
        long_content = "a" * 200
        chunk = TextChunk(content=long_content, source="big.py", start_line=1, end_line=1)
        sources = _build_sources([(chunk, 0.5)])
        assert len(sources[0].preview) == 100

    def test_preview_replaces_newlines(self) -> None:
        chunk = TextChunk(content="line1\nline2\nline3", source="f.py", start_line=1, end_line=3)
        sources = _build_sources([(chunk, 0.5)])
        assert "\n" not in sources[0].preview
        assert "line1 line2 line3" in sources[0].preview

    def test_empty_results(self) -> None:
        sources = _build_sources([])
        assert sources == []


# ---------------------------------------------------------------------------
# SourceRef and AskResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_source_ref_fields(self) -> None:
        ref = SourceRef(file="main.py", lines="1-10", relevance=0.99, preview="code here")
        assert ref.file == "main.py"
        assert ref.relevance == 0.99

    def test_ask_result_defaults(self) -> None:
        result = AskResult(answer="test")
        assert result.sources == []
        assert result.chunks_searched == 0
        assert result.total_chunks == 0

    def test_ask_result_full(self) -> None:
        result = AskResult(
            answer="The answer",
            sources=[SourceRef(file="a.py", lines="1-5", relevance=0.9, preview="abc")],
            chunks_searched=5,
            total_chunks=100,
        )
        assert len(result.sources) == 1
        assert result.chunks_searched == 5


# ---------------------------------------------------------------------------
# AskEngine init
# ---------------------------------------------------------------------------


class TestAskEngineInit:
    def test_default_config(self) -> None:
        e = AskEngine()
        assert e.ollama_url == "http://localhost:11434"
        assert e.model == "llama3.2"
        assert e.total_chunks == 0

    def test_custom_url_strips_trailing_slash(self) -> None:
        e = AskEngine(ollama_url="http://host:1234/")
        assert e.ollama_url == "http://host:1234"


# ---------------------------------------------------------------------------
# AskEngine.load
# ---------------------------------------------------------------------------


class TestAskEngineLoad:
    @pytest.mark.asyncio
    async def test_load_parses_files(self, tmp_path: Path, engine: AskEngine) -> None:
        (tmp_path / "test.txt").write_text("Hello world.\n\nSecond paragraph.\n\nThird paragraph.")
        with patch.object(engine.embeddings, "index", new_callable=AsyncMock) as mock_idx:
            await engine.load([tmp_path])
            assert engine.total_chunks > 0
            mock_idx.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_empty_dir(self, tmp_path: Path, engine: AskEngine) -> None:
        with patch.object(engine.embeddings, "index", new_callable=AsyncMock):
            await engine.load([tmp_path])
            assert engine.total_chunks == 0

    @pytest.mark.asyncio
    async def test_load_skips_unparseable_files(self, tmp_path: Path, engine: AskEngine) -> None:
        (tmp_path / "bad.bin").write_bytes(b"\x00\x01\x02")
        (tmp_path / "good.txt").write_text("some text content here\n\nmore content")
        with patch.object(engine.embeddings, "index", new_callable=AsyncMock):
            await engine.load([tmp_path])
            # Only the .txt file should have been parsed (bin not collected)
            # No exception should have been raised

    @pytest.mark.asyncio
    async def test_load_with_progress_callback(self, tmp_path: Path, engine: AskEngine) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        callback = MagicMock()
        with patch.object(engine.embeddings, "index", new_callable=AsyncMock):
            await engine.load([tmp_path], progress_callback=callback)
            callback.assert_called()
            # Callback should have been called with ("parse", current, total)
            args = callback.call_args[0]
            assert args[0] == "parse"


# ---------------------------------------------------------------------------
# AskEngine.ask (streaming)
# ---------------------------------------------------------------------------


class TestAskStreaming:
    @pytest.mark.asyncio
    async def test_ask_no_chunks_yields_no_files_message(self, engine: AskEngine) -> None:
        tokens = []
        async for token in engine.ask("What is this?"):
            tokens.append(token)
        assert "No files were loaded" in "".join(tokens)

    @pytest.mark.asyncio
    async def test_ask_no_results_yields_no_context_message(self, engine: AskEngine) -> None:
        engine._chunks = [TextChunk(content="x", source="f.py", start_line=1, end_line=1)]
        with patch.object(engine.embeddings, "search", new_callable=AsyncMock, return_value=[]):
            tokens = []
            async for token in engine.ask("question?"):
                tokens.append(token)
            assert "No relevant context found" in "".join(tokens)

    @pytest.mark.asyncio
    async def test_ask_streams_tokens(self, engine: AskEngine) -> None:
        engine._chunks = [TextChunk(content="x", source="f.py", start_line=1, end_line=1)]
        search_result = [(engine._chunks[0], 0.9)]

        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": True}),
        ]

        mock_resp = _make_stream_mock(lines)

        with (
            patch.object(
                engine.embeddings, "search", new_callable=AsyncMock, return_value=search_result
            ),
            patch.object(engine._client, "stream", return_value=mock_resp),
        ):
            tokens = []
            async for token in engine.ask("test?"):
                tokens.append(token)
            assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_ask_handles_empty_lines(self, engine: AskEngine) -> None:
        engine._chunks = [TextChunk(content="x", source="f.py", start_line=1, end_line=1)]
        search_result = [(engine._chunks[0], 0.9)]

        lines = [
            "",
            "   ",
            json.dumps({"message": {"content": "ok"}, "done": True}),
        ]

        mock_resp = _make_stream_mock(lines)

        with (
            patch.object(
                engine.embeddings, "search", new_callable=AsyncMock, return_value=search_result
            ),
            patch.object(engine._client, "stream", return_value=mock_resp),
        ):
            tokens = []
            async for token in engine.ask("test?"):
                tokens.append(token)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_ask_handles_json_decode_error(self, engine: AskEngine) -> None:
        engine._chunks = [TextChunk(content="x", source="f.py", start_line=1, end_line=1)]
        search_result = [(engine._chunks[0], 0.9)]

        lines = [
            "not valid json",
            json.dumps({"message": {"content": "ok"}, "done": True}),
        ]

        mock_resp = _make_stream_mock(lines)

        with (
            patch.object(
                engine.embeddings, "search", new_callable=AsyncMock, return_value=search_result
            ),
            patch.object(engine._client, "stream", return_value=mock_resp),
        ):
            tokens = []
            async for token in engine.ask("test?"):
                tokens.append(token)
            assert tokens == ["ok"]


# ---------------------------------------------------------------------------
# AskEngine.ask_full
# ---------------------------------------------------------------------------


class TestAskFull:
    @pytest.mark.asyncio
    async def test_ask_full_no_chunks(self, engine: AskEngine) -> None:
        result = await engine.ask_full("question?")
        assert isinstance(result, AskResult)
        assert "No files were loaded" in result.answer
        assert result.chunks_searched == 0
        assert result.total_chunks == 0

    @pytest.mark.asyncio
    async def test_ask_full_no_search_results(self, engine: AskEngine) -> None:
        engine._chunks = [TextChunk(content="x", source="f.py", start_line=1, end_line=1)]
        with patch.object(engine.embeddings, "search", new_callable=AsyncMock, return_value=[]):
            result = await engine.ask_full("question?")
            assert "No relevant context found" in result.answer
            assert result.total_chunks == 1

    @pytest.mark.asyncio
    async def test_ask_full_returns_structured_result(self, engine: AskEngine) -> None:
        chunk = TextChunk(content="def foo(): pass", source="foo.py", start_line=1, end_line=1)
        engine._chunks = [chunk]
        search_result = [(chunk, 0.85)]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "The answer is foo."}}

        with (
            patch.object(
                engine.embeddings, "search", new_callable=AsyncMock, return_value=search_result
            ),
            patch.object(engine._client, "post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            result = await engine.ask_full("What is foo?")
            assert result.answer == "The answer is foo."
            assert len(result.sources) == 1
            assert result.sources[0].file == "foo.py"
            assert result.chunks_searched == 1
            assert result.total_chunks == 1


# ---------------------------------------------------------------------------
# AskEngine.close
# ---------------------------------------------------------------------------


class TestAskEngineClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self, engine: AskEngine) -> None:
        with (
            patch.object(engine.embeddings, "close", new_callable=AsyncMock) as mock_emb_close,
            patch.object(engine._client, "aclose", new_callable=AsyncMock) as mock_client_close,
        ):
            await engine.close()
            mock_emb_close.assert_awaited_once()
            mock_client_close.assert_awaited_once()


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
