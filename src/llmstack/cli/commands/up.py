"""llmstack up — boot the full stack."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from llmstack.cli.console import console
from llmstack.config.loader import load_config
from llmstack.core.preflight import check_ports, docker_status, required_ports
from llmstack.core.stack import Stack


def _preflight(config) -> None:
    """Abort early with actionable messages instead of opaque Docker errors."""
    # 1. Docker daemon must be reachable before we try to start any container.
    docker_err = docker_status()
    if docker_err:
        console.print(Panel(docker_err, title="Docker unavailable", border_style="red"))
        raise typer.Exit(1)

    # 2. Every host port the stack binds must be free.
    conflicts = [c for c in check_ports(required_ports(config)) if not c.available]
    if not conflicts:
        return

    table = Table(show_header=True, header_style="bold red", border_style="dim")
    table.add_column("Port")
    table.add_column("Service")
    table.add_column("Held by")
    for c in conflicts:
        table.add_row(str(c.port), c.service, c.owner or "another process")

    console.print(
        Panel(
            "[error]Cannot start: required ports are already in use.[/]",
            border_style="red",
        )
    )
    console.print(table)
    console.print(
        "\n[dim]Fix it by either:[/]\n"
        "  • Stopping the process holding the port (it may be a previous run — "
        "try [bold cyan]llmstack status[/] / [bold cyan]llmstack down[/]).\n"
        "  • Changing the port in [bold]llmstack.yaml[/] and re-running.\n"
    )
    raise typer.Exit(1)


def up(
    attach: bool = typer.Option(False, "--attach", "-a", help="Stream logs after starting"),
) -> None:
    """Start all services defined in llmstack.yaml."""
    config = load_config()
    _preflight(config)
    stack = Stack(config)

    console.print("\n[bold]Starting LLMStack...[/]\n")
    asyncio.run(stack.up())

    if attach:
        console.print("[info]Streaming logs (Ctrl+C to detach)...[/]\n")
        try:
            for line in stack.docker.stream_logs("ollama", follow=True, tail=10):
                console.print(line, end="")
        except KeyboardInterrupt:
            console.print("\n[info]Detached.[/]")
