"""llmstack search — Semantic and regex code search across the codebase."""

from __future__ import annotations

import re
from pathlib import Path

from llmstack.cli.console import console


CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cpp", ".c",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala", ".dart", ".lua",
    ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".scss", ".yaml", ".yml",
    ".json", ".toml", ".md", ".txt", ".xml", ".proto",
}

IGNORE_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", "dist", "build",
               ".tox", ".eggs", ".mypy_cache", ".pytest_cache", "target", "vendor"}


def search(
    query: str,
    target: str | None = None,
    mode: str = "smart",
    file_pattern: str | None = None,
    max_results: int = 50,
    context_lines: int = 2,
    output: str | None = None,
) -> None:
    """Search code with smart matching."""
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    directory = Path(target) if target else Path.cwd()

    # Collect files
    files = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if file_pattern:
            if not p.match(file_pattern):
                continue
        elif p.suffix.lower() not in CODE_EXTS:
            continue
        files.append(p)

    console.print()
    console.print(f"[bold]llmstack search[/]  mode=[cyan]{mode}[/]  files=[dim]{len(files)}[/]")
    console.print(f'  [dim]Query: "{query}"[/]')
    console.print()

    results = []

    if mode == "regex":
        results = _search_regex(files, query, context_lines)
    elif mode == "symbol":
        results = _search_symbols(files, query)
    elif mode == "definition":
        results = _search_definitions(files, query)
    elif mode == "usage":
        results = _search_usages(files, query)
    else:  # smart
        results = _search_smart(files, query, context_lines)

    if not results:
        console.print("[dim]No results found.[/]")
        return

    # Display results
    results = results[:max_results]
    console.print(f"[bold]{len(results)} results[/]\n")

    for r in results:
        rel_path = r["file"]
        try:
            rel_path = str(Path(r["file"]).relative_to(directory))
        except ValueError:
            pass

        # Detect language for syntax highlighting
        ext = Path(r["file"]).suffix
        lang_map = {".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby"}
        lang = lang_map.get(ext, "text")

        header = f"[bold cyan]{rel_path}[/]:[bold]{r['line']}[/]"
        if r.get("symbol"):
            header += f"  [dim]{r['symbol']}[/]"

        console.print(header)
        console.print(Syntax(
            r["context"], lang, theme="monokai",
            line_numbers=True, start_line=r.get("start_line", r["line"]),
            highlight_lines={r["line"]},
        ))
        console.print()

    # Export
    if output:
        import json
        Path(output).write_text(json.dumps(results, indent=2, default=str))
        console.print(f"[green]Results saved to {output}[/]")


def _search_smart(files: list[Path], query: str, context: int) -> list[dict]:
    """Smart search: combines literal, fuzzy, and semantic matching."""
    results = []
    query_lower = query.lower()
    query_parts = query_lower.split()

    for file_path in files:
        try:
            content = file_path.read_text(errors="replace")
        except OSError:
            continue

        lines = content.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Exact match
            if query_lower in line_lower:
                score = 1.0
            # All words match
            elif all(part in line_lower for part in query_parts):
                score = 0.8
            # CamelCase/snake_case matching
            elif _flexible_match(query, line):
                score = 0.6
            else:
                continue

            start = max(0, i - context)
            end = min(len(lines), i + context + 1)

            results.append({
                "file": str(file_path),
                "line": i + 1,
                "start_line": start + 1,
                "context": "\n".join(lines[start:end]),
                "match": line.strip(),
                "score": score,
            })

    return sorted(results, key=lambda r: -r["score"])


def _flexible_match(query: str, line: str) -> bool:
    """Match CamelCase, snake_case, and partial names."""
    # Convert query to flexible pattern
    # "get user" -> matches getUserById, get_user_name, etc.
    parts = re.split(r'[\s_-]+', query.lower())
    line_lower = line.lower()
    return all(part in line_lower for part in parts)


def _search_regex(files: list[Path], pattern: str, context: int) -> list[dict]:
    """Regex search."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        console.print(f"[error]Invalid regex: {e}[/]")
        return []

    results = []
    for file_path in files:
        try:
            content = file_path.read_text(errors="replace")
        except OSError:
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                results.append({
                    "file": str(file_path),
                    "line": i + 1,
                    "start_line": start + 1,
                    "context": "\n".join(lines[start:end]),
                    "match": line.strip(),
                })

    return results


def _search_symbols(files: list[Path], query: str) -> list[dict]:
    """Search for symbol definitions (functions, classes, variables)."""
    results = []
    query_lower = query.lower()

    patterns = [
        (r'(?:def|async def)\s+(\w*{q}\w*)', "function"),
        (r'class\s+(\w*{q}\w*)', "class"),
        (r'(\w*{q}\w*)\s*=\s*', "variable"),
        (r'function\s+(\w*{q}\w*)', "function"),
        (r'(?:const|let|var)\s+(\w*{q}\w*)', "variable"),
        (r'func\s+(\w*{q}\w*)', "function"),
        (r'fn\s+(\w*{q}\w*)', "function"),
        (r'type\s+(\w*{q}\w*)', "type"),
        (r'interface\s+(\w*{q}\w*)', "interface"),
    ]

    for file_path in files:
        try:
            content = file_path.read_text(errors="replace")
        except OSError:
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines):
            for pattern_template, symbol_type in patterns:
                pattern = pattern_template.format(q=re.escape(query_lower))
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    start = max(0, i - 1)
                    end = min(len(lines), i + 5)
                    results.append({
                        "file": str(file_path),
                        "line": i + 1,
                        "start_line": start + 1,
                        "context": "\n".join(lines[start:end]),
                        "match": line.strip(),
                        "symbol": f"{symbol_type}: {match.group(1) if match.lastindex else match.group()}",
                    })
                    break

    return results


def _search_definitions(files: list[Path], query: str) -> list[dict]:
    """Search for where something is defined."""
    return _search_symbols(files, query)


def _search_usages(files: list[Path], query: str) -> list[dict]:
    """Search for where something is used (excluding definitions)."""
    results = []
    def_patterns = re.compile(
        rf'(?:def|class|function|func|fn|const|let|var|type|interface)\s+{re.escape(query)}',
        re.IGNORECASE,
    )

    for file_path in files:
        try:
            content = file_path.read_text(errors="replace")
        except OSError:
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if re.search(rf'\b{re.escape(query)}\b', line, re.IGNORECASE):
                # Exclude definitions
                if def_patterns.search(line):
                    continue
                # Exclude imports
                if line.strip().startswith(("import ", "from ")):
                    continue

                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                results.append({
                    "file": str(file_path),
                    "line": i + 1,
                    "start_line": start + 1,
                    "context": "\n".join(lines[start:end]),
                    "match": line.strip(),
                })

    return results
