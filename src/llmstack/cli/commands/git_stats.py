"""llmstack git-stats — Visualize git repository statistics."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
from collections import Counter

from llmstack.cli.console import console


def git_stats(days: int = 30, author: str | None = None) -> None:
    """Show git repository statistics."""
    from rich.table import Table
    from rich.panel import Panel

    cwd = str(Path.cwd())

    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=30,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    # Basic info
    branch = git("branch", "--show-current").strip()
    total_commits = git("rev-list", "--count", "HEAD").strip()
    first_commit = git("log", "--reverse", "--format=%ai", "-1").strip()[:10]

    # Recent commits
    since_arg = f"--since={days} days ago"
    log_args = ["log", since_arg, "--format=%H|%an|%ae|%ai|%s"]
    if author:
        log_args.extend(["--author", author])

    log_output = git(*log_args)
    commits = [line.split("|", 4) for line in log_output.strip().split("\n") if "|" in line]

    # Shortstat for recent changes
    shortstat = git("diff", "--shortstat", f"HEAD~{min(len(commits), 100)}..HEAD") if commits else ""

    console.print()
    console.print(f"[bold]llmstack git-stats[/]  branch=[cyan]{branch}[/]  period=[dim]{days} days[/]")
    console.print()

    # Summary panel
    console.print(Panel(
        f"[bold]Branch:[/] {branch}\n"
        f"[bold]Total commits:[/] {total_commits}\n"
        f"[bold]First commit:[/] {first_commit}\n"
        f"[bold]Recent commits ({days}d):[/] {len(commits)}\n"
        f"[bold]Changes:[/] {shortstat.strip() or 'N/A'}",
        title="Repository Summary",
        border_style="cyan",
    ))

    if not commits:
        console.print("[dim]No commits in the selected period.[/]")
        return

    # Author statistics
    authors = Counter()
    for c in commits:
        if len(c) >= 2:
            authors[c[1]] += 1

    if authors:
        author_table = Table(
            title="Contributors",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        author_table.add_column("Author", style="bold")
        author_table.add_column("Commits", justify="right")
        author_table.add_column("Share", justify="right")
        author_table.add_column("Activity")

        total = len(commits)
        for name, count in authors.most_common(15):
            pct = (count / total) * 100
            bar_len = min(25, int(pct / 4))
            bar = f"[green]{'█' * bar_len}{'░' * (25 - bar_len)}[/]"
            author_table.add_row(name, str(count), f"{pct:.0f}%", bar)

        console.print()
        console.print(author_table)

    # Day of week distribution
    day_counts = Counter()
    hour_counts = Counter()
    for c in commits:
        if len(c) >= 4:
            try:
                date_str = c[3]
                from datetime import datetime
                dt = datetime.fromisoformat(date_str[:19])
                day_counts[dt.strftime("%A")] += 1
                hour_counts[dt.hour] += 1
            except (ValueError, IndexError):
                pass

    if day_counts:
        console.print()
        console.print("[bold]Commits by Day[/]")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        max_day = max(day_counts.values()) or 1
        for day in day_order:
            count = day_counts.get(day, 0)
            bar_len = int((count / max_day) * 20)
            console.print(f"  {day[:3]} │ [cyan]{'█' * bar_len}[/] {count}")

    # Hour distribution
    if hour_counts:
        console.print()
        console.print("[bold]Commits by Hour[/]")
        max_hour = max(hour_counts.values()) or 1
        for h in range(24):
            count = hour_counts.get(h, 0)
            bar_len = int((count / max_hour) * 15)
            if count > 0:
                console.print(f"  {h:02d}:00 │ [cyan]{'█' * bar_len}[/] {count}")

    # File type distribution (from recent commits)
    file_types = Counter()
    changed_files = git("log", since_arg, "--name-only", "--format=").strip()
    for f in changed_files.split("\n"):
        f = f.strip()
        if f:
            ext = Path(f).suffix
            if ext:
                file_types[ext] += 1

    if file_types:
        console.print()
        type_table = Table(
            title="Files Changed by Type",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        type_table.add_column("Extension", style="bold")
        type_table.add_column("Changes", justify="right")

        for ext, count in file_types.most_common(10):
            type_table.add_row(ext, str(count))

        console.print(type_table)

    # Commit message patterns
    if commits:
        type_patterns = Counter()
        for c in commits:
            if len(c) >= 5:
                msg = c[4]
                match = re.match(r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)", msg)
                if match:
                    type_patterns[match.group(1)] += 1
                else:
                    type_patterns["other"] += 1

        if type_patterns:
            console.print()
            console.print("[bold]Commit Types[/]")
            for ctype, count in type_patterns.most_common():
                console.print(f"  {ctype:12s} │ {'█' * min(20, count)} {count}")
