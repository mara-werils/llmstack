"""llmstack dead-code — Find unused functions, classes, and imports."""

from __future__ import annotations

import json
from pathlib import Path

from llmstack.cli.console import console


def dead_code(
    target: str | None = None,
    confidence: str | None = None,
    code_type: str | None = None,
    output: str | None = None,
) -> None:
    """Find potentially dead code."""
    from rich.table import Table
    from rich.panel import Panel
    from llmstack.analyze.dead_code import DeadCodeDetector

    directory = Path(target) if target else Path.cwd()

    if not directory.is_dir():
        console.print(f"[error]Not a directory: {directory}[/]")
        return

    console.print()
    console.print(f"[bold]llmstack dead-code[/]  directory=[dim]{directory}[/]")
    console.print()

    detector = DeadCodeDetector(directory)
    items = detector.scan()

    # Filter
    if confidence:
        items = [i for i in items if i.confidence == confidence.lower()]
    if code_type:
        items = [i for i in items if i.type == code_type.lower()]

    if not items:
        console.print(Panel("[green]No dead code detected.[/]", border_style="green"))
        return

    # Group by type
    type_counts = {}
    for item in items:
        type_counts[item.type] = type_counts.get(item.type, 0) + 1

    confidence_colors = {"high": "red", "medium": "yellow", "low": "dim"}
    type_icons = {"function": "ƒ", "class": "C", "import": "→", "variable": "x"}

    table = Table(
        title=f"Potentially Dead Code ({len(items)} items)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Conf", width=6)
    table.add_column("Type", width=8)
    table.add_column("Name", style="bold")
    table.add_column("File:Line")
    table.add_column("Reason", style="dim")

    for item in items[:100]:
        color = confidence_colors.get(item.confidence, "white")
        icon = type_icons.get(item.type, "?")

        rel_file = item.file
        try:
            rel_file = str(Path(item.file).relative_to(directory))
        except ValueError:
            pass

        table.add_row(
            f"[{color}]{item.confidence}[/]",
            f"{icon} {item.type}",
            item.name,
            f"{rel_file}:{item.line}",
            item.reason[:60],
        )

    console.print(table)

    # Summary
    summary_parts = [f"{t}: {c}" for t, c in sorted(type_counts.items())]
    console.print()
    console.print(
        Panel(
            f"[bold]Total:[/] {len(items)} items\n"
            f"[bold]By type:[/] {', '.join(summary_parts)}\n\n"
            f"[dim]Note: Some items may be used dynamically (reflection, decorators, etc.)\n"
            f"Review with caution before removing.[/]",
            title="Dead Code Summary",
            border_style="yellow",
        )
    )

    if output:
        data = [
            {
                "type": i.type,
                "name": i.name,
                "file": i.file,
                "line": i.line,
                "confidence": i.confidence,
                "reason": i.reason,
            }
            for i in items
        ]
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Report saved to {output}[/]")
