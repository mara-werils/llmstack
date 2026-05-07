"""llmstack down — stop all services."""

from __future__ import annotations

import typer

from llmstack.cli.console import console
from llmstack.docker.manager import DockerManager


def down(
    volumes: bool = typer.Option(False, "--volumes", "-v", help="Also remove data volumes"),
) -> None:
    """Stop and remove all llmstack services."""
    docker = DockerManager()
    stopped = docker.stop_all(remove_volumes=volumes)

    if stopped:
        for name in stopped:
            console.print(f"  [info]Stopped {name}[/]")
        if volumes:
            console.print("[warning]Volumes removed.[/]")
        console.print("\n[success]All services stopped.[/]")
    else:
        console.print("[info]No llmstack services are running.[/]")
