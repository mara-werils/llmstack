"""Code-aware AST chunking — split code by functions and classes, not arbitrary lines.

Uses Python's built-in `ast` module for .py files and enhanced regex patterns
for other languages. Produces chunks that contain complete logical units
(a whole function, a whole class, a whole method) for better LLM understanding.

No external dependencies required.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from llmstack.ask.parsers import TextChunk


@dataclass
class CodeSymbol:
    """A code symbol (function, class, method) with its source."""

    name: str
    kind: str  # "function", "class", "method", "module_header"
    start_line: int
    end_line: int
    content: str
    decorators: list[str] | None = None
    parent: str | None = None  # parent class for methods


def chunk_python(source: str, file_path: str) -> list[TextChunk]:
    """Parse Python source using the AST for precise function/class boundaries."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to line-based chunking
        return _fallback_chunk(source, file_path)

    lines = source.splitlines()
    symbols: list[CodeSymbol] = []

    # Collect module-level docstring + imports as header
    header_end = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_end = max(header_end, node.end_lineno or node.lineno)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module docstring
            header_end = max(header_end, node.end_lineno or node.lineno)

    if header_end > 0:
        symbols.append(
            CodeSymbol(
                name="<module_header>",
                kind="module_header",
                start_line=1,
                end_line=header_end,
                content="\n".join(lines[:header_end]),
            )
        )

    # Collect top-level functions and classes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or node.lineno
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind="function",
                    start_line=node.lineno,
                    end_line=end,
                    content="\n".join(lines[node.lineno - 1 : end]),
                    decorators=[
                        "\n".join(lines[d.lineno - 1 : (d.end_lineno or d.lineno)])
                        for d in node.decorator_list
                    ],
                )
            )

        elif isinstance(node, ast.ClassDef):
            end = node.end_lineno or node.lineno
            # Add the full class as one chunk
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind="class",
                    start_line=node.lineno,
                    end_line=end,
                    content="\n".join(lines[node.lineno - 1 : end]),
                )
            )

            # Also add individual methods for large classes
            if end - node.lineno > 50:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        child_end = child.end_lineno or child.lineno
                        symbols.append(
                            CodeSymbol(
                                name=f"{node.name}.{child.name}",
                                kind="method",
                                start_line=child.lineno,
                                end_line=child_end,
                                content="\n".join(lines[child.lineno - 1 : child_end]),
                                parent=node.name,
                            )
                        )

    if not symbols:
        return _fallback_chunk(source, file_path)

    # Convert to TextChunks — deduplicate overlapping ranges
    chunks: list[TextChunk] = []
    seen_ranges: set[tuple[int, int]] = set()

    for sym in symbols:
        key = (sym.start_line, sym.end_line)
        if key in seen_ranges:
            continue

        # Skip methods if the whole class is already small enough
        if sym.kind == "method":
            parent_sym = next(
                (s for s in symbols if s.kind == "class" and s.name == sym.parent),
                None,
            )
            if parent_sym and (parent_sym.end_line - parent_sym.start_line) <= 50:
                continue

        seen_ranges.add(key)
        header = f"# {sym.kind}: {sym.name}\n" if sym.kind != "module_header" else ""
        chunks.append(
            TextChunk(
                content=header + sym.content,
                source=file_path,
                start_line=sym.start_line,
                end_line=sym.end_line,
            )
        )

    # Capture any lines not covered by symbols (module-level code between functions)
    all_covered = set()
    for sym in symbols:
        for ln in range(sym.start_line, sym.end_line + 1):
            all_covered.add(ln)

    uncovered_lines = []
    uncovered_start = None
    for i, line in enumerate(lines, 1):
        if i not in all_covered and line.strip():
            if uncovered_start is None:
                uncovered_start = i
            uncovered_lines.append(line)
        else:
            if uncovered_lines and len(uncovered_lines) >= 3:
                chunks.append(
                    TextChunk(
                        content="\n".join(uncovered_lines),
                        source=file_path,
                        start_line=uncovered_start,
                        end_line=i - 1,
                    )
                )
            uncovered_lines = []
            uncovered_start = None

    if uncovered_lines and len(uncovered_lines) >= 3:
        chunks.append(
            TextChunk(
                content="\n".join(uncovered_lines),
                source=file_path,
                start_line=uncovered_start,
                end_line=len(lines),
            )
        )

    chunks.sort(key=lambda c: c.start_line)
    return chunks


# ---------------------------------------------------------------------------
# Language-specific boundary patterns for non-Python languages
# ---------------------------------------------------------------------------

_LANG_PATTERNS: dict[str, re.Pattern] = {
    "js": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\()",
        re.MULTILINE,
    ),
    "ts": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|interface\s+\w+|type\s+\w+|const\s+\w+\s*=)",
        re.MULTILINE,
    ),
    "go": re.compile(r"^(?:func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+|type\s+\w+\s+struct)", re.MULTILINE),
    "rs": re.compile(
        r"^(?:pub\s+)?(?:fn\s+\w+|struct\s+\w+|impl\s+|enum\s+\w+|trait\s+\w+)", re.MULTILINE
    ),
    "java": re.compile(
        r"^\s*(?:public|private|protected|static|\s)*(?:class\s+\w+|interface\s+\w+|\w+\s+\w+\s*\()",
        re.MULTILINE,
    ),
    "c": re.compile(r"^(?:\w+[\s\*]+\w+\s*\([^;]*$|struct\s+\w+|typedef\s+)", re.MULTILINE),
    "rb": re.compile(r"^(?:def\s+\w+|class\s+\w+|module\s+\w+)", re.MULTILINE),
}

_EXT_TO_LANG: dict[str, str] = {
    ".js": "js",
    ".jsx": "js",
    ".mjs": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".go": "go",
    ".rs": "rs",
    ".java": "java",
    ".c": "c",
    ".cpp": "c",
    ".h": "c",
    ".hpp": "c",
    ".rb": "rb",
}


def chunk_code(source: str, file_path: str) -> list[TextChunk]:
    """Smart code chunking — uses AST for Python, regex boundaries for others."""
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        return chunk_python(source, file_path)

    lang = _EXT_TO_LANG.get(ext)
    if lang and lang in _LANG_PATTERNS:
        return _chunk_by_pattern(source, file_path, _LANG_PATTERNS[lang])

    return _fallback_chunk(source, file_path)


def _chunk_by_pattern(source: str, file_path: str, pattern: re.Pattern) -> list[TextChunk]:
    """Split source code at regex-matched boundaries."""
    lines = source.splitlines()
    matches = list(pattern.finditer(source))

    if len(matches) < 2:
        return _fallback_chunk(source, file_path)

    # Find line numbers for each match
    boundaries: list[int] = []
    for m in matches:
        line_num = source[: m.start()].count("\n") + 1
        boundaries.append(line_num)

    chunks: list[TextChunk] = []

    # Header (before first boundary)
    if boundaries[0] > 1:
        header_lines = lines[: boundaries[0] - 1]
        if any(ln.strip() for ln in header_lines):
            chunks.append(
                TextChunk(
                    content="\n".join(header_lines),
                    source=file_path,
                    start_line=1,
                    end_line=boundaries[0] - 1,
                )
            )

    # Chunks between boundaries
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else len(lines)
        chunk_lines = lines[start - 1 : end]
        if any(ln.strip() for ln in chunk_lines):
            chunks.append(
                TextChunk(
                    content="\n".join(chunk_lines),
                    source=file_path,
                    start_line=start,
                    end_line=end,
                )
            )

    return chunks if chunks else _fallback_chunk(source, file_path)


def _fallback_chunk(source: str, file_path: str, max_lines: int = 80) -> list[TextChunk]:
    """Fallback: split into fixed-size line blocks."""
    lines = source.splitlines()
    if not lines:
        return []

    chunks: list[TextChunk] = []
    for i in range(0, len(lines), max_lines):
        block = lines[i : i + max_lines]
        if any(ln.strip() for ln in block):
            chunks.append(
                TextChunk(
                    content="\n".join(block),
                    source=file_path,
                    start_line=i + 1,
                    end_line=min(i + max_lines, len(lines)),
                )
            )
    return chunks
