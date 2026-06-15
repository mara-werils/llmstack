"""llmstack verify-private — prove the stack runs 100% on your machine.

Audits llmstack.yaml for anything that could send code or prompts off the
machine (cloud providers, webhooks, network-capable agent tools, open CORS).
Exits non-zero when the local-only guarantee is broken, so it can gate CI.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from llmstack.cli.console import console
from llmstack.core.privacy import CRITICAL, INFO, WARNING, audit_privacy

_SEV_STYLE = {CRITICAL: "bold red", WARNING: "yellow", INFO: "cyan"}
_SEV_ICON = {CRITICAL: "✖", WARNING: "⚠", INFO: "ℹ"}


def verify_private(
    target: str | None = None,
    json_output: bool = False,
) -> None:
    """Verify that the configured stack keeps all data local."""
    from rich.panel import Panel
    from rich.table import Table

    from llmstack.config.loader import config_exists, load_config

    directory = Path(target) if target else Path.cwd()

    if not config_exists(directory):
        console.print(
            f"[red]No llmstack.yaml found in {directory}.[/] Run 'llmstack init' first."
        )
        raise SystemExit(2)

    config = load_config(directory)
    report = audit_privacy(config)

    if json_output:
        console.print_json(_json.dumps(report.to_dict()))
        raise SystemExit(0 if report.is_private else 1)

    console.print()
    console.print("[bold]llmstack verify-private[/]  — local-only guarantee audit")
    console.print()

    if report.findings:
        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Sev", width=11)
        table.add_column("Area")
        table.add_column("Finding")
        table.add_column("Recommendation", style="dim")
        for f in sorted(report.findings, key=lambda x: (x.severity != CRITICAL, x.severity)):
            style = _SEV_STYLE.get(f.severity, "white")
            icon = _SEV_ICON.get(f.severity, "•")
            table.add_row(
                f"[{style}]{icon} {f.severity}[/]", f.category, f.detail, f.recommendation
            )
        console.print(table)
        console.print()

    color = "green" if report.is_private else "red"
    console.print(
        Panel(
            f"[bold]Verdict:[/] [{color}]{report.verdict}[/]\n"
            f"[bold]Critical:[/] {len(report.critical)}    "
            f"[bold]Warnings:[/] {len(report.warnings)}\n\n"
            + (
                "[green]Code and prompts never leave this machine.[/]"
                if report.is_private
                else "[red]This configuration can send data to external services.[/]"
            ),
            title="Privacy Guarantee",
            border_style=color,
        )
    )

    raise SystemExit(0 if report.is_private else 1)
