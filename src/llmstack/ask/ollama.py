"""Ollama connectivity and model provisioning for `llmstack ask`.

Centralizes the first-run experience so a new user reaches their first answer
with zero guesswork:

* :func:`check_ollama` distinguishes "not installed" from "installed but not
  running" and returns a platform-aware install hint.
* :func:`ensure_models` verifies every required model up front and downloads the
  missing ones with a *real* streaming progress bar — no more silent multi-minute
  hangs, and no more late 404s after the index has already been built.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass

import httpx

from llmstack.cli.console import console


@dataclass
class OllamaStatus:
    """Result of probing the local Ollama daemon."""

    reachable: bool
    installed: bool
    version: str | None = None


async def check_ollama(url: str, timeout: float = 5.0) -> OllamaStatus:
    """Probe Ollama at ``url`` and report whether it is installed and reachable."""
    url = url.rstrip("/")
    installed = shutil.which("ollama") is not None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/api/version")
            if resp.status_code == 200:
                version = resp.json().get("version")
                # If the daemon answers, Ollama is effectively installed even if
                # the binary lives outside PATH (e.g. the macOS app bundle).
                return OllamaStatus(reachable=True, installed=True, version=version)
            return OllamaStatus(reachable=False, installed=installed)
    except httpx.HTTPError:
        return OllamaStatus(reachable=False, installed=installed)


def install_hint(status: OllamaStatus, url: str) -> str:
    """Build an actionable, platform-aware message for an unreachable daemon."""
    if status.installed:
        return (
            "[error]Ollama is installed but not running.[/]\n\n"
            "Start it with:\n  [bold cyan]ollama serve[/]\n\n"
            f"[dim]Tried: {url}[/]"
        )

    if sys.platform == "darwin":
        install = "[bold cyan]brew install ollama[/]   [dim]# or download from https://ollama.com/download[/]"
    elif sys.platform.startswith("linux"):
        install = "[bold cyan]curl -fsSL https://ollama.com/install.sh | sh[/]"
    else:  # win32 and anything else
        install = "Download the installer from [bold cyan]https://ollama.com/download[/]"

    return (
        "[error]Ollama is not installed.[/]\n\n"
        "llmstack runs models locally through Ollama. Install it:\n"
        f"  {install}\n\n"
        "Then start it and re-run your command:\n  [bold cyan]ollama serve[/]"
    )


async def _model_exists(client: httpx.AsyncClient, url: str, model: str) -> bool:
    """Return True if ``model`` is already present in the local Ollama store."""
    try:
        resp = await client.post(f"{url}/api/show", json={"name": model})
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _pull_with_progress(client: httpx.AsyncClient, url: str, model: str) -> None:
    """Download ``model`` via ``/api/pull`` rendering live per-layer progress."""
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    console.print(
        f"  [cyan]Downloading model [bold]{model}[/][/] [dim](first run only — this is cached)[/]"
    )

    with Progress(
        TextColumn("  [blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Ollama streams one progress object per layer (keyed by digest), so we
        # keep a task per digest and update its byte count as it arrives.
        tasks: dict[str, int] = {}
        async with client.stream(
            "POST",
            f"{url}/api/pull",
            json={"name": model, "stream": True},
            timeout=httpx.Timeout(None, connect=10),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("error"):
                    raise RuntimeError(data["error"])

                digest = data.get("digest")
                total = data.get("total")
                completed = data.get("completed", 0)

                if digest and total:
                    if digest not in tasks:
                        label = digest.split(":")[-1][:12]
                        tasks[digest] = progress.add_task(label, total=total)
                    progress.update(tasks[digest], completed=completed)


async def ensure_models(
    url: str,
    models: list[str],
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Ensure each model in ``models`` is available locally, pulling the missing ones.

    Probes for presence first so the common "everything already cached" path is
    instant and silent. Missing models are downloaded once with a progress bar.
    De-duplicates ``models`` while preserving order.
    """
    url = url.rstrip("/")
    seen: set[str] = set()
    unique = [m for m in models if m and not (m in seen or seen.add(m))]

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10))
    try:
        for model in unique:
            if await _model_exists(client, url, model):
                continue
            await _pull_with_progress(client, url, model)
    finally:
        if owns_client:
            await client.aclose()
