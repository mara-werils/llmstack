"""llmstack up — boot the full stack."""

from __future__ import annotations

import asyncio

import typer

from llmstack.cli.console import console
from llmstack.config.loader import load_config
from llmstack.core.stack import Stack


def up(
    attach: bool = typer.Option(False, "--attach", "-a", help="Stream logs after starting"),
) -> None:
    """Start all services defined in llmstack.yaml."""
    config = load_config()
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
