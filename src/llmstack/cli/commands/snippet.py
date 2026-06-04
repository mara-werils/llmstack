"""llmstack snippet — Save, search, and reuse code snippets."""

from __future__ import annotations

from pathlib import Path

from llmstack.cli.console import console


EXTENSION_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
}


def snippet_save(
    file: str | None = None,
    title: str | None = None,
    tags: str | None = None,
    description: str = "",
    lines: str | None = None,
) -> None:
    """Save a code snippet from a file or stdin."""
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()

    if file:
        file_path = Path(file)
        if not file_path.exists():
            console.print(f"[error]File not found: {file}[/]")
            return

        content = file_path.read_text(errors="replace")
        language = EXTENSION_TO_LANG.get(file_path.suffix.lower(), "")

        # Extract specific lines if requested
        if lines:
            all_lines = content.split("\n")
            parts = lines.split("-")
            start = max(0, int(parts[0]) - 1)
            end = int(parts[1]) if len(parts) > 1 else start + 1
            content = "\n".join(all_lines[start:end])

        if not title:
            title = file_path.name
            if lines:
                title += f" (L{lines})"

        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        snippet = mgr.save(
            title=title,
            code=content,
            language=language,
            tags=tag_list,
            description=description,
            source_file=str(file_path),
        )
        console.print(f"[green]Snippet saved:[/] [bold]{snippet.title}[/]  id=[dim]{snippet.id}[/]")
    else:
        console.print("[error]Provide a file with --file or pipe content to stdin[/]")


def snippet_search(
    query: str = "",
    language: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> None:
    """Search saved snippets."""
    from rich.table import Table
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()
    results = mgr.search(query=query, language=language, tag=tag, limit=limit)

    if not results:
        console.print("[dim]No snippets found.[/]")
        if not query and not tag and not language:
            console.print("[dim]Save snippets with: llmstack snippet save --file <path>[/]")
        return

    table = Table(
        title=f"Snippets ({len(results)} found)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("ID", width=12)
    table.add_column("Title", style="bold")
    table.add_column("Language", width=12)
    table.add_column("Tags")
    table.add_column("Uses", justify="right", width=5)

    for s in results:
        table.add_row(
            s.id,
            s.title,
            s.language or "-",
            ", ".join(s.tags) if s.tags else "-",
            str(s.usage_count),
        )

    console.print(table)


def snippet_show(snippet_id: str) -> None:
    """Show a specific snippet."""
    from rich.syntax import Syntax
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()
    snippet = mgr.get(snippet_id)

    if not snippet:
        console.print(f"[error]Snippet not found: {snippet_id}[/]")
        return

    console.print()
    console.print(f"[bold]{snippet.title}[/]  id=[dim]{snippet.id}[/]")
    if snippet.description:
        console.print(f"  [dim]{snippet.description}[/]")
    if snippet.tags:
        console.print(f"  Tags: {', '.join(snippet.tags)}")
    if snippet.source_file:
        console.print(f"  Source: [dim]{snippet.source_file}[/]")
    console.print()

    lang = snippet.language or "text"
    console.print(Syntax(snippet.code, lang, theme="monokai", line_numbers=True))


def snippet_delete(snippet_id: str) -> None:
    """Delete a snippet."""
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()
    if mgr.delete(snippet_id):
        console.print(f"[green]Snippet {snippet_id} deleted.[/]")
    else:
        console.print(f"[error]Snippet not found: {snippet_id}[/]")


def snippet_tags() -> None:
    """List all tags."""
    from rich.table import Table
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()
    tags = mgr.list_tags()

    if not tags:
        console.print("[dim]No tags found.[/]")
        return

    table = Table(title="Snippet Tags", header_style="bold cyan", border_style="dim")
    table.add_column("Tag", style="bold")
    table.add_column("Count", justify="right")

    for tag, count in tags.items():
        table.add_row(tag, str(count))

    console.print(table)


def snippet_export(output: str | None = None) -> None:
    """Export all snippets."""
    import json
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()
    data = mgr.export_all()

    if output:
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Exported {len(data)} snippets to {output}[/]")
    else:
        console.print(json.dumps(data, indent=2))


def snippet_stats() -> None:
    """Show snippet statistics."""
    from rich.panel import Panel
    from llmstack.snippets.manager import SnippetManager

    mgr = SnippetManager()
    total = mgr.count()
    languages = mgr.list_languages()
    tags = mgr.list_tags()

    lang_str = ", ".join(f"{k}: {v}" for k, v in list(languages.items())[:10])
    tag_str = ", ".join(f"{k} ({v})" for k, v in list(tags.items())[:10])

    console.print(
        Panel(
            f"[bold]Total snippets:[/] {total}\n"
            f"[bold]Languages:[/] {lang_str or 'none'}\n"
            f"[bold]Top tags:[/] {tag_str or 'none'}",
            title="Snippet Library",
            border_style="cyan",
        )
    )
