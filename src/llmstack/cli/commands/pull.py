"""llmstack pull — pull models from Ollama with progress display."""

from __future__ import annotations

import json
import sys

import httpx

from llmstack.cli.console import console


def pull(model: str, ollama_url: str = "http://localhost:11434") -> None:
    """Pull a model from Ollama registry with progress feedback."""
    console.print(f"[accent]Pulling model:[/] [model]{model}[/]")

    try:
        httpx.get(f"{ollama_url}/api/tags", timeout=5)
    except (httpx.ConnectError, httpx.TimeoutException):
        console.print("[error]Cannot connect to Ollama. Is it running?[/]")
        console.print("[muted]  Install: https://ollama.com/download[/]")
        sys.exit(1)

    from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Pulling {model}", total=None)

        try:
            with httpx.stream(
                "POST",
                f"{ollama_url}/api/pull",
                json={"name": model, "stream": True},
                timeout=httpx.Timeout(600, connect=10),
            ) as resp:
                if resp.status_code != 200:
                    console.print(f"[error]Pull failed: HTTP {resp.status_code}[/]")
                    sys.exit(1)

                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    status = data.get("status", "")
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)

                    if total > 0:
                        progress.update(task_id, total=total, completed=completed, description=status)
                    else:
                        progress.update(task_id, description=status)

                    if data.get("error"):
                        console.print(f"[error]{data['error']}[/]")
                        sys.exit(1)

        except httpx.ConnectError:
            console.print("[error]Connection lost during pull.[/]")
            sys.exit(1)

    console.print(f"[success]\u2713[/] Model [model]{model}[/] is ready")
