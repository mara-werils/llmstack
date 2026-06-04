"""File content extraction and chunking for llmstack ask."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextChunk:
    """A chunk of text extracted from a file."""

    content: str
    source: str  # file path
    start_line: int  # for citations
    end_line: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAIN_TEXT_EXTS = {".txt", ".md", ".rst"}
CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".rb",
    ".php",
    ".swift",
    ".kt",
}
CONFIG_EXTS = {".json", ".yaml", ".yml", ".toml"}
MARKUP_EXTS = {".html", ".xml"}
SUPPORTED_EXTS = (
    PLAIN_TEXT_EXTS
    | CODE_EXTS
    | CONFIG_EXTS
    | MARKUP_EXTS
    | {
        ".csv",
        ".log",
        ".pdf",
        ".docx",
    }
)

SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    "venv",
    ".venv",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "egg-info",
}

# Regex patterns for detecting function/class boundaries in code
_CODE_BOUNDARY = re.compile(
    r"^(?:"
    r"(?:export\s+)?(?:async\s+)?(?:def |class |function |func |fn |pub\s+fn |"
    r"pub\s+(?:async\s+)?fn |impl |struct |enum |trait |interface |"
    r"(?:public|private|protected)\s+(?:static\s+)?(?:class|void|int|String|"
    r"boolean|async)\s+\w+|"
    r"module |object |case\s+class )"
    r")",
    re.MULTILINE,
)

_TIMESTAMP_PATTERN = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}")

_HTML_TAG = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_files(path: Path) -> list[Path]:
    """Recursively collect all supported files under *path*.

    Skips hidden directories and well-known junk directories.
    If *path* is a file, returns it in a single-element list (if supported).
    """
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTS:
            return [path]
        return []

    results: list[Path] = []
    _walk(path, results)
    results.sort()
    return results


def parse_file(path: Path) -> list[TextChunk]:
    """Parse a file into text chunks based on its type."""
    path = path.resolve()
    ext = path.suffix.lower()

    if ext in PLAIN_TEXT_EXTS:
        return _parse_plain_text(path)
    if ext in CODE_EXTS:
        return _parse_code(path)
    if ext in CONFIG_EXTS:
        return _parse_config(path)
    if ext == ".csv":
        return _parse_csv(path)
    if ext in MARKUP_EXTS:
        return _parse_markup(path)
    if ext == ".log":
        return _parse_log(path)
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)

    # Fallback: try reading as plain text
    return _parse_plain_text(path)


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------


def _walk(directory: Path, results: list[Path]) -> None:
    """Walk *directory* recursively, appending supported files to *results*."""
    try:
        entries = sorted(directory.iterdir())
    except PermissionError:
        return

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            _walk(entry, results)
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTS:
            results.append(entry)


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Read a file as UTF-8 text, replacing errors."""
    return path.read_text(encoding="utf-8", errors="replace")


def _lines_to_chunks(lines: list[str], source: str, max_lines: int = 100) -> list[TextChunk]:
    """Split a list of lines into chunks of at most *max_lines*."""
    chunks: list[TextChunk] = []
    for i in range(0, len(lines), max_lines):
        block = lines[i : i + max_lines]
        content = "\n".join(block).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    source=source,
                    start_line=i + 1,
                    end_line=i + len(block),
                )
            )
    return chunks


def _parse_plain_text(path: Path) -> list[TextChunk]:
    """Split plain text by paragraphs (double newlines)."""
    text = _read_text(path)
    source = str(path)
    lines = text.splitlines()

    if not lines:
        return []

    chunks: list[TextChunk] = []
    para_lines: list[str] = []
    para_start = 1

    for i, line in enumerate(lines, start=1):
        if line.strip() == "" and para_lines:
            content = "\n".join(para_lines).strip()
            if content:
                chunks.append(
                    TextChunk(
                        content=content,
                        source=source,
                        start_line=para_start,
                        end_line=i - 1,
                    )
                )
            para_lines = []
            para_start = i + 1
        else:
            if not para_lines:
                para_start = i
            para_lines.append(line)

    # Final paragraph
    if para_lines:
        content = "\n".join(para_lines).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    source=source,
                    start_line=para_start,
                    end_line=len(lines),
                )
            )

    return (
        chunks
        if chunks
        else [
            TextChunk(
                content=text.strip(), source=source, start_line=1, end_line=max(len(lines), 1)
            )
        ]
    )


def _parse_code(path: Path) -> list[TextChunk]:
    """Split code files by function/class boundaries or ~100 line blocks."""
    text = _read_text(path)
    source = str(path)
    lines = text.splitlines()

    if not lines:
        return []

    # Find boundary line numbers
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if _CODE_BOUNDARY.match(line):
            boundaries.append(i)

    if len(boundaries) < 2:
        # Few boundaries — fall back to fixed-size blocks
        return _lines_to_chunks(lines, source, max_lines=100)

    chunks: list[TextChunk] = []

    # Content before first boundary
    if boundaries[0] > 0:
        pre = "\n".join(lines[: boundaries[0]]).strip()
        if pre:
            chunks.append(
                TextChunk(
                    content=pre,
                    source=source,
                    start_line=1,
                    end_line=boundaries[0],
                )
            )

    # Each boundary to the next
    for idx in range(len(boundaries)):
        start = boundaries[idx]
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    source=source,
                    start_line=start + 1,
                    end_line=end,
                )
            )

    return chunks if chunks else _lines_to_chunks(lines, source, max_lines=100)


def _parse_config(path: Path) -> list[TextChunk]:
    """Parse config files. Small files as single chunk, large ones split by top-level keys."""
    text = _read_text(path)
    source = str(path)
    lines = text.splitlines()

    if len(lines) <= 50:
        return (
            [
                TextChunk(
                    content=text.strip(),
                    source=source,
                    start_line=1,
                    end_line=max(len(lines), 1),
                )
            ]
            if text.strip()
            else []
        )

    # For large config files, split by top-level keys (lines with no leading whitespace)
    ext = path.suffix.lower()
    if ext == ".json":
        return _lines_to_chunks(lines, source, max_lines=80)

    # YAML / TOML: top-level keys start at column 0 and are not comments
    chunks: list[TextChunk] = []
    block_lines: list[str] = []
    block_start = 1

    for i, line in enumerate(lines, start=1):
        is_top_level = (
            line
            and not line[0].isspace()
            and not line.startswith("#")
            and not line.startswith("---")
        )
        if is_top_level and block_lines:
            content = "\n".join(block_lines).strip()
            if content:
                chunks.append(
                    TextChunk(
                        content=content,
                        source=source,
                        start_line=block_start,
                        end_line=i - 1,
                    )
                )
            block_lines = [line]
            block_start = i
        else:
            if not block_lines:
                block_start = i
            block_lines.append(line)

    if block_lines:
        content = "\n".join(block_lines).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    source=source,
                    start_line=block_start,
                    end_line=len(lines),
                )
            )

    return (
        chunks
        if chunks
        else [
            TextChunk(
                content=text.strip(), source=source, start_line=1, end_line=max(len(lines), 1)
            )
        ]
    )


def _parse_csv(path: Path) -> list[TextChunk]:
    """Read first 100 rows of a CSV as text."""
    text = _read_text(path)
    source = str(path)
    reader = csv.reader(io.StringIO(text))
    rows: list[str] = []
    for i, row in enumerate(reader):
        if i >= 100:
            break
        rows.append(",".join(row))

    content = "\n".join(rows).strip()
    if not content:
        return []
    return [
        TextChunk(
            content=content,
            source=source,
            start_line=1,
            end_line=len(rows),
        )
    ]


def _parse_markup(path: Path) -> list[TextChunk]:
    """Strip HTML/XML tags and extract text content."""
    text = _read_text(path)
    source = str(path)

    # Remove script and style blocks
    text_clean = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text_clean = re.sub(r"<style[^>]*>.*?</style>", "", text_clean, flags=re.DOTALL | re.IGNORECASE)
    # Strip tags
    text_clean = _HTML_TAG.sub("", text_clean)
    # Collapse whitespace
    text_clean = re.sub(r"\n{3,}", "\n\n", text_clean).strip()

    if not text_clean:
        return []

    lines = text_clean.splitlines()
    return _lines_to_chunks(lines, source, max_lines=80)


def _parse_log(path: Path) -> list[TextChunk]:
    """Split log files by timestamp patterns or every 50 lines."""
    text = _read_text(path)
    source = str(path)
    lines = text.splitlines()

    if not lines:
        return []

    # Try timestamp-based splitting first
    chunks: list[TextChunk] = []
    block_lines: list[str] = []
    block_start = 1

    timestamp_count = sum(1 for ln in lines[:100] if _TIMESTAMP_PATTERN.match(ln))
    use_timestamps = timestamp_count > 5  # enough timestamps to be meaningful

    if use_timestamps:
        for i, line in enumerate(lines, start=1):
            if _TIMESTAMP_PATTERN.match(line) and block_lines and len(block_lines) >= 10:
                content = "\n".join(block_lines).strip()
                if content:
                    chunks.append(
                        TextChunk(
                            content=content,
                            source=source,
                            start_line=block_start,
                            end_line=i - 1,
                        )
                    )
                block_lines = [line]
                block_start = i
            else:
                if not block_lines:
                    block_start = i
                block_lines.append(line)

        if block_lines:
            content = "\n".join(block_lines).strip()
            if content:
                chunks.append(
                    TextChunk(
                        content=content,
                        source=source,
                        start_line=block_start,
                        end_line=len(lines),
                    )
                )

        if chunks:
            return chunks

    # Fallback: every 50 lines
    return _lines_to_chunks(lines, source, max_lines=50)


def _parse_pdf(path: Path) -> list[TextChunk]:
    """Parse PDF files using pymupdf (fitz). Graceful fallback if not installed."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return [
            TextChunk(
                content="[PDF support unavailable — pip install pymupdf for PDF support]",
                source=str(path),
                start_line=1,
                end_line=1,
            )
        ]

    source = str(path)
    chunks: list[TextChunk] = []

    try:
        doc = fitz.open(str(path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                chunks.append(
                    TextChunk(
                        content=text,
                        source=source,
                        start_line=page_num + 1,
                        end_line=page_num + 1,
                    )
                )
        doc.close()
    except Exception as exc:
        chunks.append(
            TextChunk(
                content=f"[Error reading PDF: {exc}]",
                source=source,
                start_line=1,
                end_line=1,
            )
        )

    return chunks


def _parse_docx(path: Path) -> list[TextChunk]:
    """Parse DOCX files using python-docx. Graceful fallback if not installed."""
    try:
        import docx  # python-docx
    except ImportError:
        return [
            TextChunk(
                content="[DOCX support unavailable — pip install python-docx for DOCX support]",
                source=str(path),
                start_line=1,
                end_line=1,
            )
        ]

    source = str(path)
    try:
        document = docx.Document(str(path))
        paragraphs: list[str] = []
        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        if not paragraphs:
            return []

        # Group paragraphs into reasonable chunks
        chunks: list[TextChunk] = []
        block: list[str] = []
        block_start = 1

        for i, para in enumerate(paragraphs, start=1):
            block.append(para)
            if len(block) >= 10:
                chunks.append(
                    TextChunk(
                        content="\n\n".join(block),
                        source=source,
                        start_line=block_start,
                        end_line=i,
                    )
                )
                block = []
                block_start = i + 1

        if block:
            chunks.append(
                TextChunk(
                    content="\n\n".join(block),
                    source=source,
                    start_line=block_start,
                    end_line=len(paragraphs),
                )
            )

        return chunks

    except Exception as exc:
        return [
            TextChunk(
                content=f"[Error reading DOCX: {exc}]",
                source=source,
                start_line=1,
                end_line=1,
            )
        ]
