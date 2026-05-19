"""llmstack history — view and search ask conversation history."""

from __future__ import annotations

from pathlib import Path

from rich.table import Table

from llmstack.cli.console import console, banner


def history(
    index_dir: str | None = None,
    limit: int = 20,
    search: str | None = None,
) -> None:
    """Show conversation history from the persistent ask index."""
    idx_dir = Path(index_dir) if index_dir else Path.cwd() / ".llmstack-index"

    if not idx_dir.exists():
        console.print("[muted]No conversation history found.[/]")
        console.print("[muted]Run 'llmstack ask' to start building history.[/]")
        return

    conv_file = idx_dir / "conversations.jsonl"
    if not conv_file.exists():
        console.print("[muted]No conversations recorded yet.[/]")
        return

    import json

    conversations = []
    for line in conv_file.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            conversations.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if search:
        search_lower = search.lower()
        conversations = [
            c for c in conversations
            if search_lower in c.get("question", "").lower()
            or search_lower in c.get("answer", "").lower()
        ]

    conversations = conversations[-limit:]

    banner("Ask History", f"{len(conversations)} conversation(s)")

    if not conversations:
        console.print("\n[muted]No matching conversations found.[/]\n")
        return

    table = Table(show_header=True, show_edge=False)
    table.add_column("#", style="muted", width=4)
    table.add_column("Question", style="highlight", max_width=50)
    table.add_column("Sources", style="muted")
    table.add_column("Time", style="muted")

    for i, conv in enumerate(conversations, 1):
        question = conv.get("question", "")[:50]
        sources = conv.get("sources_count", len(conv.get("sources", [])))
        timestamp = conv.get("timestamp", "")
        if "T" in str(timestamp):
            timestamp = str(timestamp).split("T")[0]

        table.add_row(str(i), question, str(sources), timestamp)

    console.print(table)
    console.print()
