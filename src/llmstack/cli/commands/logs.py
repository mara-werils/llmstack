"""llmstack logs — stream logs from a service."""

from __future__ import annotations

import typer

from llmstack.cli.console import console
from llmstack.docker.manager import DockerManager


def logs(
    service: str = typer.Argument(help="Service name (e.g., ollama, qdrant, redis)"),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f", help="Follow log output"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of lines to show"),
) -> None:
    """Stream logs from a specific service."""
    docker = DockerManager()
    try:
        for line in docker.stream_logs(service, follow=follow, tail=tail):
            console.print(line, end="", highlight=False)
    except ValueError as exc:
        console.print(f"[error]{exc}[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        pass
