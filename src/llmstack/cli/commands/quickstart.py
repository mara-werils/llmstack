"""llmstack quickstart — zero-to-running in one command."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx

from llmstack.cli.console import console, banner, success, failure, warn


def quickstart(
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    skip_pull: bool = False,
) -> None:
    """Get from zero to a running LLMStack in one command."""
    banner("LLMStack Quickstart", "Zero to running in one command")
    console.print()

    # Step 1: Check prerequisites
    console.print("[accent]Step 1/4[/] Checking prerequisites...")

    if shutil.which("docker"):
        success("Docker installed")
    else:
        failure("Docker is not installed")
        console.print("  [muted]Install: https://docs.docker.com/get-docker/[/]")
        sys.exit(1)

    # Step 2: Check/start Ollama
    console.print(f"\n[accent]Step 2/4[/] Checking Ollama at {ollama_url}...")

    ollama_running = False
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            ollama_running = True
            success("Ollama is running")
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    if not ollama_running:
        warn("Ollama is not reachable")
        console.print("  [muted]Install Ollama: https://ollama.com/download[/]")
        console.print("  [muted]Then run: ollama serve[/]")
        sys.exit(1)

    # Step 3: Pull model if needed
    console.print(f"\n[accent]Step 3/4[/] Ensuring model '{model}' is available...")

    if not skip_pull:
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
            available = [m.get("name", "") for m in resp.json().get("models", [])]
            # Check both exact and prefix match
            has_model = any(
                name == model or name.startswith(f"{model}:")
                for name in available
            )
            if has_model:
                success(f"Model '{model}' is available")
            else:
                warn(f"Model '{model}' not found locally, pulling...")
                from llmstack.cli.commands.pull import pull
                pull(model=model, ollama_url=ollama_url)
        except Exception as exc:
            failure(f"Model check failed: {exc}")

    # Step 4: Create config if needed
    console.print("\n[accent]Step 4/4[/] Setting up configuration...")

    config_path = Path.cwd() / "llmstack.yaml"
    if config_path.exists():
        success("llmstack.yaml already exists")
    else:
        from llmstack.cli.commands.init import init
        init(preset="chat")
        success("Created llmstack.yaml with 'chat' preset")

    console.print()
    banner("Ready!", "Run 'llmstack ask -i .' to chat with your codebase")
    console.print()
    console.print("  [muted]Or try:[/]")
    console.print("  [highlight]llmstack ask 'How does auth work?' ./src/[/]")
    console.print("  [highlight]llmstack chat[/]")
    console.print("  [highlight]llmstack up[/]  (start full gateway stack)")
    console.print("  [highlight]llmstack playground[/]  (open Web UI)")
    console.print()
