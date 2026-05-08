"""Detailed parser tests for llmstack ask file content extraction."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from llmstack.ask.parsers import (
    TextChunk,
    collect_files,
    parse_file,
    SUPPORTED_EXTS,
    SKIP_DIRS,
    _parse_plain_text,
    _parse_code,
    _parse_config,
    _parse_csv,
    _parse_markup,
    _parse_log,
    _lines_to_chunks,
)


# ---------------------------------------------------------------------------
# Plain text splitting
# ---------------------------------------------------------------------------


class TestPlainTextParsing:
    """Tests for plain text paragraph splitting."""

    def test_split_by_paragraphs(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
        chunks = parse_file(f)
        assert len(chunks) == 3
        assert chunks[0].content == "First paragraph."
        assert chunks[1].content == "Second paragraph."
        assert chunks[2].content == "Third paragraph."

    def test_single_paragraph(self, tmp_path: Path) -> None:
        f = tmp_path / "single.txt"
        f.write_text("Just one paragraph with no blank lines.")
        chunks = parse_file(f)
        assert len(chunks) == 1
        assert chunks[0].content == "Just one paragraph with no blank lines."

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        chunks = parse_file(f)
        assert len(chunks) == 0

    def test_line_numbers_correct(self, tmp_path: Path) -> None:
        f = tmp_path / "numbered.txt"
        f.write_text("Line one.\nLine two.\n\nLine four.\nLine five.")
        chunks = parse_file(f)
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2
        assert chunks[1].start_line == 4

    def test_rst_treated_as_plain_text(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.rst"
        f.write_text("Title\n=====\n\nSome content.")
        chunks = parse_file(f)
        assert len(chunks) >= 1

    def test_multiple_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "spaced.txt"
        f.write_text("A\n\n\n\nB")
        chunks = parse_file(f)
        assert len(chunks) == 2
        assert chunks[0].content == "A"
        assert chunks[1].content == "B"


# ---------------------------------------------------------------------------
# Python code splitting
# ---------------------------------------------------------------------------


class TestPythonCodeParsing:
    """Tests for Python code splitting by functions/classes."""

    def test_split_by_functions(self, tmp_path: Path) -> None:
        f = tmp_path / "funcs.py"
        f.write_text(textwrap.dedent("""\
            import os

            def func_a():
                pass

            def func_b():
                pass

            class MyClass:
                def method(self):
                    pass
        """))
        chunks = parse_file(f)
        assert len(chunks) >= 3  # import block + func_a + func_b/class

    def test_small_file_stays_together(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny.py"
        f.write_text("x = 1\ny = 2\n")
        chunks = parse_file(f)
        assert len(chunks) == 1

    def test_source_path_stored(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("def hello(): pass")
        chunks = parse_file(f)
        assert chunks[0].source == str(f.resolve())


# ---------------------------------------------------------------------------
# JSON / YAML / TOML parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Tests for config file parsing."""

    def test_small_json_single_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "small.json"
        data = {"name": "test", "version": "1.0"}
        f.write_text(json.dumps(data, indent=2))
        chunks = parse_file(f)
        assert len(chunks) == 1
        assert "name" in chunks[0].content

    def test_yaml_parsing(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        f.write_text("key: value\nlist:\n  - a\n  - b\n")
        chunks = parse_file(f)
        assert len(chunks) >= 1
        assert "key" in chunks[0].content

    def test_toml_parsing(self, tmp_path: Path) -> None:
        f = tmp_path / "config.toml"
        f.write_text('[tool]\nname = "test"\n')
        chunks = parse_file(f)
        assert len(chunks) >= 1

    def test_large_yaml_split_by_keys(self, tmp_path: Path) -> None:
        """YAML files over 50 lines should be split by top-level keys."""
        f = tmp_path / "big.yaml"
        lines = []
        for i in range(10):
            lines.append(f"key_{i}:")
            for j in range(6):
                lines.append(f"  sub_{j}: value_{j}")
        f.write_text("\n".join(lines))
        chunks = parse_file(f)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# HTML / XML parsing
# ---------------------------------------------------------------------------


class TestMarkupParsing:
    """Tests for HTML/XML tag stripping."""

    def test_strip_html_tags(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Hello</p><p>World</p></body></html>")
        chunks = parse_file(f)
        content = " ".join(c.content for c in chunks)
        assert "Hello" in content
        assert "<p>" not in content

    def test_strip_script_tags(self, tmp_path: Path) -> None:
        f = tmp_path / "script.html"
        f.write_text(
            "<html><script>var x=1;</script><body>Content here</body></html>"
        )
        chunks = parse_file(f)
        content = " ".join(c.content for c in chunks)
        assert "Content here" in content
        assert "var x" not in content

    def test_xml_parsing(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xml"
        f.write_text("<root><item>Value</item></root>")
        chunks = parse_file(f)
        content = " ".join(c.content for c in chunks)
        assert "Value" in content


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


class TestCsvParsing:
    """Tests for CSV row limiting."""

    def test_csv_reads_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        rows = ["name,age"] + [f"person_{i},{i}" for i in range(50)]
        f.write_text("\n".join(rows))
        chunks = parse_file(f)
        assert len(chunks) == 1
        assert "person_0" in chunks[0].content

    def test_csv_limits_to_100_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "big.csv"
        rows = ["col"] + [f"row_{i}" for i in range(200)]
        f.write_text("\n".join(rows))
        chunks = parse_file(f)
        assert len(chunks) == 1
        assert "row_99" not in chunks[0].content or "row_150" not in chunks[0].content
        # Verify it's capped — content should have at most ~100 lines
        line_count = chunks[0].content.count("\n") + 1
        assert line_count <= 100


# ---------------------------------------------------------------------------
# Log file parsing
# ---------------------------------------------------------------------------


class TestLogParsing:
    """Tests for log file parsing."""

    def test_log_split_by_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "app.log"
        lines = [f"Log line {i}" for i in range(120)]
        f.write_text("\n".join(lines))
        chunks = parse_file(f)
        assert len(chunks) >= 2  # 120 lines / 50 per chunk

    def test_log_with_timestamps(self, tmp_path: Path) -> None:
        f = tmp_path / "timed.log"
        lines = []
        for i in range(30):
            lines.append(f"2024-01-{i+1:02d}T10:00:00 Event {i}")
        f.write_text("\n".join(lines))
        chunks = parse_file(f)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Graceful fallback tests
# ---------------------------------------------------------------------------


class TestGracefulFallbacks:
    """Tests for unsupported format handling."""

    def test_pdf_without_pymupdf(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        # This should not crash even without pymupdf installed
        chunks = parse_file(f)
        assert len(chunks) >= 1

    def test_docx_without_python_docx(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.docx"
        f.write_bytes(b"PK\x03\x04 fake docx")
        chunks = parse_file(f)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Large file handling
# ---------------------------------------------------------------------------


class TestLargeFileHandling:
    """Tests for handling large files gracefully."""

    def test_large_text_file_chunked(self, tmp_path: Path) -> None:
        f = tmp_path / "large.txt"
        paragraphs = [f"Paragraph {i} content here." for i in range(100)]
        f.write_text("\n\n".join(paragraphs))
        chunks = parse_file(f)
        assert len(chunks) == 100

    def test_large_code_file_chunked(self, tmp_path: Path) -> None:
        f = tmp_path / "large.py"
        funcs = [f"def func_{i}():\n    pass\n\n" for i in range(50)]
        f.write_text("\n".join(funcs))
        chunks = parse_file(f)
        assert len(chunks) >= 10

    def test_lines_to_chunks_helper(self) -> None:
        lines = [f"line {i}" for i in range(250)]
        chunks = _lines_to_chunks(lines, "test.txt", max_lines=100)
        assert len(chunks) == 3
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 100
        assert chunks[1].start_line == 101


# ---------------------------------------------------------------------------
# Collect files edge cases
# ---------------------------------------------------------------------------


class TestCollectFilesEdgeCases:
    """Edge case tests for file collection."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        files = collect_files(tmp_path)
        assert files == []

    def test_nested_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "nested.py").write_text("x = 1")
        files = collect_files(tmp_path)
        assert len(files) == 1
        assert "nested.py" in files[0].name

    def test_skip_venv(self, tmp_path: Path) -> None:
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "activate.py").write_text("# activate")
        (tmp_path / "app.py").write_text("# app")
        files = collect_files(tmp_path)
        assert len(files) == 1
        assert "app.py" in files[0].name
