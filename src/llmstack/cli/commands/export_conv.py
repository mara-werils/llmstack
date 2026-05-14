"""llmstack export-conv — export conversation history."""

from __future__ import annotations

import json
from pathlib import Path

from llmstack.cli.console import console


def export_conv(
    output: str | None = None,
    format: str = "markdown",
    index_dir: str | None = None,
) -> None:
    """Export conversation history from persistent index."""
    index_path = Path(index_dir) if index_dir else Path.cwd() / ".llmstack-index"
    conv_file = index_path / "conversation.json"

    if not conv_file.exists():
        console.print("[warning]No conversation history found.[/]")
        console.print(f"[dim]Expected at: {conv_file}[/]")
        return

    try:
        data = json.loads(conv_file.read_text())
    except Exception as e:
        console.print(f"[error]Failed to read conversation: {e}[/]")
        return

    messages = data.get("messages", []) if isinstance(data, dict) else data

    if format == "json":
        out = json.dumps({"messages": messages}, indent=2)
        ext = "json"
    else:
        lines = ["# Conversation Export\n"]
        for msg in messages:
            role = msg.get("role", "").title()
            content = msg.get("content", "")
            lines.append(f"## {role}\n\n{content}\n")
        out = "\n".join(lines)
        ext = "md"

    if output:
        out_path = Path(output)
    else:
        out_path = Path.cwd() / f"conversation.{ext}"

    out_path.write_text(out)
    console.print(f"[green]Conversation exported to {out_path}[/]")
    console.print(f"  [dim]{len(messages)} messages[/]")
