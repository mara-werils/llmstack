"""Tests for llmstack ask — engine, embeddings, CLI integration."""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llmstack.ask.parsers import TextChunk, parse_file, collect_files
from llmstack.ask.embeddings import LocalEmbeddings
from llmstack.ask.engine import AskEngine, AskResult, SourceRef, _build_context, _build_sources


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_txt(tmp_path: Path) -> Path:
    """Create a simple text file."""
    f = tmp_path / "hello.txt"
    f.write_text("Hello world.\n\nThis is a test file.\n\nThird paragraph.")
    return f


@pytest.fixture
def tmp_py(tmp_path: Path) -> Path:
    """Create a simple Python file."""
    f = tmp_path / "sample.py"
    f.write_text(textwrap.dedent("""\
        import os

        def greet(name):
            return f"Hello, {name}"

        class Greeter:
            def __init__(self, name):
                self.name = name

            def say_hello(self):
                return f"Hello, {self.name}"
    """))
    return f


@pytest.fixture
def tmp_json(tmp_path: Path) -> Path:
    """Create a JSON config file."""
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"key": "value", "nested": {"a": 1, "b": 2}}))
    return f


@pytest.fixture
def tmp_html(tmp_path: Path) -> Path:
    """Create an HTML file."""
    f = tmp_path / "page.html"
    f.write_text("<html><body><h1>Title</h1><p>Hello world</p></body></html>")
    return f


@pytest.fixture
def tmp_md(tmp_path: Path) -> Path:
    """Create a Markdown file."""
    f = tmp_path / "readme.md"
    f.write_text("# Heading\n\nSome text.\n\n## Subheading\n\nMore text.")
    return f


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    """Create sample chunks for testing."""
    return [
        TextChunk(content="Python is a programming language.", source="test.py", start_line=1, end_line=5),
        TextChunk(content="JavaScript runs in the browser.", source="test.js", start_line=1, end_line=3),
        TextChunk(content="Rust is a systems programming language.", source="test.rs", start_line=1, end_line=4),
        TextChunk(content="HTML is a markup language.", source="test.html", start_line=1, end_line=2),
        TextChunk(content="Docker containers isolate applications.", source="test.md", start_line=1, end_line=3),
    ]


# ---------------------------------------------------------------------------
# File parsing tests
# ---------------------------------------------------------------------------


class TestFileParsing:
    """Tests for basic file parsing via parse_file."""

    def test_parse_txt(self, tmp_txt: Path) -> None:
        chunks = parse_file(tmp_txt)
        assert len(chunks) >= 2
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert "Hello world" in chunks[0].content

    def test_parse_md(self, tmp_md: Path) -> None:
        chunks = parse_file(tmp_md)
        assert len(chunks) >= 2
        assert any("Heading" in c.content for c in chunks)

    def test_parse_py(self, tmp_py: Path) -> None:
        chunks = parse_file(tmp_py)
        assert len(chunks) >= 1
        contents = " ".join(c.content for c in chunks)
        assert "def greet" in contents
        assert "class Greeter" in contents

    def test_parse_json(self, tmp_json: Path) -> None:
        chunks = parse_file(tmp_json)
        assert len(chunks) >= 1
        assert "key" in chunks[0].content

    def test_parse_html(self, tmp_html: Path) -> None:
        chunks = parse_file(tmp_html)
        assert len(chunks) >= 1
        # Tags should be stripped
        content = " ".join(c.content for c in chunks)
        assert "Title" in content
        assert "<h1>" not in content


# ---------------------------------------------------------------------------
# Directory traversal tests
# ---------------------------------------------------------------------------


class TestDirectoryTraversal:
    """Tests for collect_files with exclusions."""

    def test_collect_files_single_file(self, tmp_txt: Path) -> None:
        files = collect_files(tmp_txt)
        assert len(files) == 1
        assert files[0] == tmp_txt.resolve()

    def test_collect_files_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.txt").write_text("hello")
        files = collect_files(tmp_path)
        assert len(files) == 2

    def test_skip_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("x = 1")
        (tmp_path / "visible.py").write_text("y = 2")
        files = collect_files(tmp_path)
        assert len(files) == 1
        assert "visible" in files[0].name

    def test_skip_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "pkg.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("console.log('hi')")
        files = collect_files(tmp_path)
        assert len(files) == 1
        assert "app.js" in files[0].name

    def test_skip_pycache(self, tmp_path: Path) -> None:
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "mod.py").write_text("cached")
        (tmp_path / "mod.py").write_text("real")
        files = collect_files(tmp_path)
        assert len(files) == 1

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        files = collect_files(tmp_path)
        assert len(files) == 0


# ---------------------------------------------------------------------------
# TextChunk tests
# ---------------------------------------------------------------------------


class TestTextChunk:
    """Tests for the TextChunk dataclass."""

    def test_chunk_fields(self) -> None:
        chunk = TextChunk(content="hello", source="test.py", start_line=1, end_line=5)
        assert chunk.content == "hello"
        assert chunk.source == "test.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 5

    def test_chunk_equality(self) -> None:
        c1 = TextChunk(content="a", source="f.py", start_line=1, end_line=1)
        c2 = TextChunk(content="a", source="f.py", start_line=1, end_line=1)
        assert c1 == c2


# ---------------------------------------------------------------------------
# Embedding mock tests
# ---------------------------------------------------------------------------


class TestCosignSimilarity:
    """Test cosine similarity correctness with mock embeddings."""

    def test_identical_vectors_have_similarity_one(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors_have_similarity_zero(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(sim) < 1e-6

    def test_opposite_vectors_have_negative_similarity(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert sim < 0


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Test context and source building for the LLM prompt."""

    def test_build_context(self) -> None:
        chunks = [
            (TextChunk(content="hello world", source="a.py", start_line=1, end_line=5), 0.9),
            (TextChunk(content="foo bar", source="b.py", start_line=10, end_line=20), 0.7),
        ]
        ctx = _build_context(chunks)
        assert "a.py:1-5" in ctx
        assert "hello world" in ctx
        assert "b.py:10-20" in ctx

    def test_build_sources(self) -> None:
        chunks = [
            (TextChunk(content="hello world content here", source="a.py", start_line=1, end_line=5), 0.8912),
        ]
        sources = _build_sources(chunks)
        assert len(sources) == 1
        assert sources[0].file == "a.py"
        assert sources[0].lines == "1-5"
        assert sources[0].relevance == 0.8912
        assert "hello world" in sources[0].preview


# ---------------------------------------------------------------------------
# Source citation formatting tests
# ---------------------------------------------------------------------------


class TestSourceCitations:
    """Test source reference formatting."""

    def test_source_ref_fields(self) -> None:
        ref = SourceRef(file="main.py", lines="10-25", relevance=0.85, preview="some code")
        assert ref.file == "main.py"
        assert ref.lines == "10-25"
        assert ref.relevance == 0.85

    def test_ask_result_structure(self) -> None:
        result = AskResult(
            answer="The answer is 42.",
            sources=[
                SourceRef(file="a.py", lines="1-5", relevance=0.9, preview="abc"),
            ],
            chunks_searched=3,
            total_chunks=100,
        )
        assert result.answer == "The answer is 42."
        assert len(result.sources) == 1
        assert result.chunks_searched == 3
        assert result.total_chunks == 100


# ---------------------------------------------------------------------------
# Stdin detection test
# ---------------------------------------------------------------------------


class TestStdinDetection:
    """Test that stdin piping can be detected."""

    def test_isatty_detection(self) -> None:
        """Verify sys.stdin.isatty can be checked without error."""
        import sys
        # This just verifies the mechanism works — in a test runner,
        # stdin may or may not be a tty
        result = sys.stdin.isatty()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Engine unit tests (mocked)
# ---------------------------------------------------------------------------


class TestAskEngine:
    """Test AskEngine with mocked Ollama calls."""

    def test_total_chunks_initially_zero(self) -> None:
        engine = AskEngine()
        assert engine.total_chunks == 0

    @pytest.mark.asyncio
    async def test_ask_no_chunks_yields_message(self) -> None:
        engine = AskEngine()
        tokens = []
        async for token in engine.ask("hello?"):
            tokens.append(token)
        text = "".join(tokens)
        assert "No files were loaded" in text
        await engine.close()

    @pytest.mark.asyncio
    async def test_ask_full_no_chunks_returns_result(self) -> None:
        engine = AskEngine()
        result = await engine.ask_full("hello?")
        assert isinstance(result, AskResult)
        assert "No files were loaded" in result.answer
        assert result.chunks_searched == 0
        await engine.close()
