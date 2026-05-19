"""Rich console singleton and display helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.theme import Theme

theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "header": "bold magenta",
    "highlight": "bold cyan",
    "muted": "dim",
    "accent": "bold blue",
    "path": "underline cyan",
    "model": "bold green",
    "cost": "bold yellow",
    "speed": "bold magenta",
})

console = Console(theme=theme)


def banner(title: str, subtitle: str = "") -> None:
    """Print a styled banner with optional subtitle."""
    text = f"[header]{title}[/]"
    if subtitle:
        text += f"\n[muted]{subtitle}[/]"
    console.print(Panel(text, border_style="blue", padding=(0, 2)))


def success(message: str) -> None:
    """Print a success message with checkmark."""
    console.print(f"  [success]\u2713[/] {message}")


def failure(message: str) -> None:
    """Print a failure message with cross."""
    console.print(f"  [error]\u2717[/] {message}")


def warn(message: str) -> None:
    """Print a warning message."""
    console.print(f"  [warning]\u26a0[/] {message}")


def info(message: str) -> None:
    """Print an info message."""
    console.print(f"  [info]\u2139[/] {message}")


@contextmanager
def spinner(message: str) -> Iterator[Progress]:
    """Show a spinner while a long operation is in progress."""
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(message, total=None)
        yield progress


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Context manager that prints elapsed time on exit."""
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    console.print(f"  [muted]{label}: {elapsed:.2f}s[/]")
