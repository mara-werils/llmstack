"""llmstack tokens — Count tokens and manage context windows."""

from __future__ import annotations

import re
from pathlib import Path

from llmstack.cli.console import console


# Approximate token counts per character for different content types
# These are rough estimates; actual tokenization varies by model
CHARS_PER_TOKEN = 4  # Average English text
CODE_CHARS_PER_TOKEN = 3.5  # Code tends to have shorter tokens


# Model context windows
MODEL_CONTEXTS = {
    "llama3.2": 131072,
    "llama3.2:1b": 131072,
    "llama3.2:3b": 131072,
    "llama3.1": 131072,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama3": 8192,
    "mistral": 32768,
    "mixtral": 32768,
    "codellama": 16384,
    "deepseek-coder": 16384,
    "deepseek-coder-v2": 131072,
    "phi3": 131072,
    "phi3:mini": 131072,
    "gemma2": 8192,
    "gemma2:9b": 8192,
    "qwen2": 32768,
    "qwen2.5": 131072,
    "command-r": 131072,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
}


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """Estimate token count for text."""
    chars_per_tok = CODE_CHARS_PER_TOKEN if is_code else CHARS_PER_TOKEN
    return max(1, int(len(text) / chars_per_tok))


def count_tokens_file(file_path: Path) -> dict:
    """Count tokens for a single file."""
    try:
        content = file_path.read_text(errors="replace")
    except OSError:
        return {"file": str(file_path), "tokens": 0, "chars": 0, "lines": 0, "error": True}

    code_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".rb", ".php"}
    is_code = file_path.suffix.lower() in code_exts

    tokens = estimate_tokens(content, is_code=is_code)
    lines = content.count("\n") + 1
    words = len(content.split())

    return {
        "file": str(file_path),
        "tokens": tokens,
        "chars": len(content),
        "lines": lines,
        "words": words,
        "is_code": is_code,
    }


def tokens(
    target: str | None = None,
    model: str = "llama3.2",
    recursive: bool = True,
    show_files: bool = True,
) -> None:
    """Count tokens in files and check context window fit."""
    from rich.table import Table
    from rich.panel import Panel

    target_path = Path(target) if target else Path.cwd()
    context_window = MODEL_CONTEXTS.get(model, 8192)

    ignore_dirs = {"__pycache__", ".git", "node_modules", "venv", ".venv", "dist", "build", ".tox"}
    code_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".rb", ".php",
                 ".swift", ".kt", ".scala", ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
                 ".html", ".css", ".sql", ".sh", ".bash"}

    if target_path.is_file():
        files = [target_path]
    elif recursive:
        files = [
            p for p in sorted(target_path.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in code_exts
            and not any(part in ignore_dirs for part in p.parts)
        ]
    else:
        files = [
            p for p in sorted(target_path.iterdir())
            if p.is_file() and p.suffix.lower() in code_exts
        ]

    if not files:
        console.print("[warning]No files found.[/]")
        return

    results = [count_tokens_file(f) for f in files]
    total_tokens = sum(r["tokens"] for r in results)
    total_chars = sum(r["chars"] for r in results)
    total_lines = sum(r["lines"] for r in results)
    total_words = sum(r["words"] for r in results)

    console.print()
    console.print(f"[bold]llmstack tokens[/]  model=[cyan]{model}[/]  context=[dim]{context_window:,}[/]")
    console.print()

    if show_files and len(results) > 1:
        table = Table(
            title=f"Token Analysis ({len(results)} files)",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("File", style="bold")
        table.add_column("Tokens", justify="right")
        table.add_column("Lines", justify="right")
        table.add_column("% of Context", justify="right")
        table.add_column("Bar", width=20)

        # Sort by token count descending
        for r in sorted(results, key=lambda x: -x["tokens"])[:50]:
            pct = (r["tokens"] / context_window) * 100
            bar_len = min(20, int(pct / 5))
            bar_color = "green" if pct < 25 else "yellow" if pct < 75 else "red"
            bar = f"[{bar_color}]{'█' * bar_len}{'░' * (20 - bar_len)}[/]"

            rel_path = str(Path(r["file"]).relative_to(target_path)) if not target_path.is_file() else r["file"]
            table.add_row(
                rel_path[:40],
                f"{r['tokens']:,}",
                f"{r['lines']:,}",
                f"{pct:.1f}%",
                bar,
            )

        console.print(table)

    # Summary
    fit_pct = (total_tokens / context_window) * 100
    fit_color = "green" if fit_pct < 50 else "yellow" if fit_pct < 100 else "red"
    fits = "YES" if total_tokens <= context_window else "NO"
    fits_color = "green" if fits == "YES" else "red"

    remaining = max(0, context_window - total_tokens)

    console.print()
    console.print(Panel(
        f"[bold]Total tokens:[/] {total_tokens:,}\n"
        f"[bold]Total chars:[/]  {total_chars:,}\n"
        f"[bold]Total lines:[/]  {total_lines:,}\n"
        f"[bold]Total words:[/]  {total_words:,}\n"
        f"[bold]Files:[/]        {len(results)}\n\n"
        f"[bold]Model:[/]        {model}\n"
        f"[bold]Context:[/]      {context_window:,} tokens\n"
        f"[bold]Used:[/]         [{fit_color}]{fit_pct:.1f}%[/]\n"
        f"[bold]Remaining:[/]    {remaining:,} tokens\n"
        f"[bold]Fits in context:[/] [{fits_color}]{fits}[/]",
        title="Token Summary",
        border_style=fit_color,
    ))

    # Recommendations
    if total_tokens > context_window:
        console.print()
        console.print("[yellow]Recommendations to reduce tokens:[/]")
        # Find largest files
        largest = sorted(results, key=lambda x: -x["tokens"])[:3]
        for r in largest:
            console.print(f"  • Consider splitting: {r['file']} ({r['tokens']:,} tokens)")
        console.print(f"  • Use --top-k to limit context in llmstack ask")
        console.print(f"  • Try a model with larger context (e.g., llama3.1 = 128K)")
