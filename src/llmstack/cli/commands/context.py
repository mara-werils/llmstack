"""llmstack context — Build optimized context for LLM prompts."""

from __future__ import annotations

from pathlib import Path

from llmstack.cli.console import console


def context(
    query: str,
    target: str | None = None,
    strategy: str = "smart",
    max_tokens: int = 8000,
    output: str | None = None,
    copy: bool = False,
) -> None:
    """Build optimized context from codebase for a query."""
    from rich.table import Table
    from llmstack.context.builder import ContextBuilder

    directory = Path(target) if target else Path.cwd()
    builder = ContextBuilder(directory, max_tokens=max_tokens)
    chunks = builder.build(query, strategy=strategy)

    if not chunks:
        console.print("[warning]No relevant context found.[/]")
        return

    console.print()
    console.print(f"[bold]llmstack context[/]  strategy=[cyan]{strategy}[/]  budget=[dim]{max_tokens} tokens[/]")
    console.print(f"  [dim]Query: {query}[/]")
    console.print()

    # Summary table
    table = Table(
        title=f"Context Chunks ({len(chunks)} selected)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("File", style="bold")
    table.add_column("Lines")
    table.add_column("Tokens", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Reason", style="dim")

    total_tokens = 0
    for chunk in chunks:
        rel_bar = "█" * int(chunk.relevance * 10)
        table.add_row(
            chunk.file[:40],
            f"{chunk.line_start}-{chunk.line_end}",
            str(chunk.tokens_estimate),
            f"[{'green' if chunk.relevance > 0.5 else 'yellow'}]{rel_bar}[/] {chunk.relevance:.2f}",
            chunk.reason,
        )
        total_tokens += chunk.tokens_estimate

    console.print(table)
    console.print(f"\n[bold]Total tokens:[/] {total_tokens:,} / {max_tokens:,}")

    # Build combined context
    combined = ""
    for chunk in chunks:
        combined += f"\n--- {chunk.file} (L{chunk.line_start}-{chunk.line_end}) ---\n"
        combined += chunk.content + "\n"

    if output:
        Path(output).write_text(combined)
        console.print(f"[green]Context saved to {output}[/]")
    elif copy:
        try:
            import subprocess
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            process.communicate(combined.encode())
            console.print("[green]Context copied to clipboard![/]")
        except Exception:
            console.print("[warning]Could not copy to clipboard. Use --output instead.[/]")
