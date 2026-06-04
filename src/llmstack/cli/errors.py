"""Actionable CLI error handler -- turns cryptic errors into helpful suggestions."""

from __future__ import annotations

import functools
import re
import sys
from typing import Callable

from llmstack.cli.console import console


# Map of known error patterns to actionable suggestions
_ERROR_SUGGESTIONS: list[tuple[str, str, str]] = [
    (
        "Connection refused",
        "Ollama is not running",
        "Start Ollama with: ollama serve\n  Or start the full stack: llmstack up",
    ),
    (
        "ConnectError",
        "Cannot connect to the inference backend",
        "Check if Ollama is running: curl http://localhost:11434\n  Or start it with: llmstack up",
    ),
    (
        "model.*not found",
        "Model not found",
        "Pull the model first: ollama pull <model_name>\n"
        "  Or run: llmstack quickstart --model llama3.2",
    ),
    (
        "Address already in use",
        "Port conflict",
        "Another process is using the port.\n"
        "  Find it: lsof -i :8000\n"
        "  Or use a different port: llmstack serve --port 8001",
    ),
    (
        "docker.errors",
        "Docker error",
        "Check Docker is running: docker info\n  Or install: https://docs.docker.com/get-docker/",
    ),
    (
        "redis.exceptions.ConnectionError",
        "Cannot connect to Redis",
        "Start Redis: docker run -d -p 6379:6379 redis:7\n  Or start the full stack: llmstack up",
    ),
    (
        "No such file or directory.*llmstack.yaml",
        "Configuration file not found",
        "Create one with: llmstack init\n  Or quickstart: llmstack quickstart",
    ),
    (
        "PermissionError",
        "Permission denied",
        "Check file permissions or run with appropriate privileges.\n"
        "  If Docker-related: ensure your user is in the 'docker' group.",
    ),
    (
        "CUDA out of memory",
        "GPU out of memory",
        "Try a smaller model: llmstack chat --model llama3.2:1b\n"
        "  Or reduce context: --max-tokens 512",
    ),
    (
        "ModuleNotFoundError",
        "Missing Python dependency",
        "Install the required extras: pip install llmstack-cli[all]\n"
        "  Or for gateway only: pip install llmstack-cli[gateway]",
    ),
]


def friendly_errors(func: Callable) -> Callable:
    """Decorator that catches exceptions and shows actionable error messages."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        try:
            return func(*args, **kwargs)
        except SystemExit:
            raise
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/]")
            sys.exit(130)
        except Exception as exc:
            error_str = str(exc)
            error_type = type(exc).__name__

            # Search for matching suggestion
            for pattern, title, suggestion in _ERROR_SUGGESTIONS:
                if re.search(pattern, error_str, re.IGNORECASE) or re.search(
                    pattern, error_type, re.IGNORECASE
                ):
                    console.print(f"\n[bold red]Error:[/] {title}")
                    console.print(f"  [dim]{error_str}[/]\n")
                    console.print(f"[bold]How to fix:[/]\n  {suggestion}\n")
                    console.print("[dim]Run 'llmstack doctor' for a full system check.[/]")
                    sys.exit(1)

            # No match -- show raw error with general advice
            console.print(f"\n[bold red]Error:[/] {error_type}: {error_str}")
            console.print("\n[dim]Run 'llmstack doctor' to diagnose common issues.[/]")
            sys.exit(1)

    return wrapper
