"""llmstack analytics — View usage statistics and trends."""

from __future__ import annotations

from llmstack.cli.console import console


def analytics(days: int = 30, output: str | None = None) -> None:
    """Show usage analytics dashboard."""
    import json
    from pathlib import Path
    from rich.table import Table
    from rich.panel import Panel
    from llmstack.analytics.tracker import AnalyticsTracker

    tracker = AnalyticsTracker()
    summary = tracker.get_summary(days=days)
    streak = tracker.get_streak()

    console.print()
    console.print(f"[bold]llmstack analytics[/]  period=[dim]{days} days[/]")
    console.print()

    # Key metrics
    total = summary["total_requests"]
    tokens = summary["total_tokens"]
    avg_dur = summary["avg_duration"]
    success = summary["success_rate"]

    metrics_panel = Panel(
        f"[bold]Total Requests:[/]  {total:,}\n"
        f"[bold]Total Tokens:[/]    {tokens:,}\n"
        f"[bold]Avg Duration:[/]    {avg_dur:.2f}s\n"
        f"[bold]Success Rate:[/]    {success:.1f}%\n"
        f"[bold]Usage Streak:[/]    {streak} days 🔥" if streak > 0 else
        f"[bold]Total Requests:[/]  {total:,}\n"
        f"[bold]Total Tokens:[/]    {tokens:,}\n"
        f"[bold]Avg Duration:[/]    {avg_dur:.2f}s\n"
        f"[bold]Success Rate:[/]    {success:.1f}%",
        title="Key Metrics",
        border_style="cyan",
    )
    console.print(metrics_panel)

    # Commands table
    if summary["by_command"]:
        cmd_table = Table(
            title="Usage by Command",
            show_header=True, header_style="bold cyan", border_style="dim",
        )
        cmd_table.add_column("Command", style="bold")
        cmd_table.add_column("Count", justify="right")
        cmd_table.add_column("Tokens", justify="right")
        cmd_table.add_column("Time", justify="right")

        for cmd in summary["by_command"][:15]:
            cmd_table.add_row(
                cmd["command"],
                str(cmd["count"]),
                f"{cmd.get('tokens', 0):,}",
                f"{cmd.get('duration', 0):.1f}s",
            )
        console.print()
        console.print(cmd_table)

    # Models table
    if summary["by_model"]:
        model_table = Table(
            title="Usage by Model",
            show_header=True, header_style="bold cyan", border_style="dim",
        )
        model_table.add_column("Model", style="bold")
        model_table.add_column("Count", justify="right")
        model_table.add_column("Avg Duration", justify="right")
        model_table.add_column("Tokens", justify="right")

        for m in summary["by_model"][:10]:
            model_table.add_row(
                m["model"],
                str(m["count"]),
                f"{m.get('avg_duration', 0):.2f}s",
                f"{m.get('tokens', 0):,}",
            )
        console.print()
        console.print(model_table)

    # Activity chart (simple ASCII)
    if summary["daily_trend"]:
        console.print()
        max_count = max(d["count"] for d in summary["daily_trend"]) or 1
        console.print("[bold]Daily Activity[/]")
        for day_data in summary["daily_trend"][-14:]:  # Last 14 days
            bar_len = int((day_data["count"] / max_count) * 30)
            bar = "█" * bar_len
            console.print(f"  {day_data['day'][-5:]} │ [green]{bar}[/] {day_data['count']}")

    if not total:
        console.print()
        console.print("[dim]No usage data yet. Start using llmstack commands to see analytics![/]")

    if output:
        Path(output).write_text(json.dumps(summary, indent=2))
        console.print(f"\n[green]Analytics exported to {output}[/]")
