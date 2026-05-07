"""llmstack doctor — diagnose common issues."""

from __future__ import annotations

import shutil
import socket

from llmstack.cli.console import console
from llmstack.core.hardware import detect_hardware


def _check_port(port: int) -> bool:
    """Return True if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def doctor() -> None:
    """Check system requirements and diagnose common issues."""
    console.print("\n[bold]LLMStack Doctor[/]\n")
    issues = 0

    # Docker
    if shutil.which("docker"):
        console.print("  [green]PASS[/] Docker is installed")
    else:
        console.print("  [red]FAIL[/] Docker is not installed")
        issues += 1

    # Docker daemon
    try:
        import docker
        client = docker.from_env()
        client.ping()
        console.print("  [green]PASS[/] Docker daemon is running")
    except Exception:
        console.print("  [red]FAIL[/] Docker daemon is not reachable")
        issues += 1

    # Hardware
    hw = detect_hardware()
    if hw.gpu_vendor != "none":
        console.print(f"  [green]PASS[/] GPU detected: {hw.gpu_name}")
        if hw.gpu_vendor == "nvidia" and hw.docker_runtime != "nvidia":
            console.print("  [yellow]WARN[/] nvidia-container-toolkit not found (GPU passthrough may not work)")
    else:
        console.print("  [yellow]WARN[/] No GPU detected (CPU inference only)")

    console.print(f"  [green]INFO[/] RAM: {hw.ram_mb // 1024} GB, CPU: {hw.cpu_cores} cores")

    # Ports
    for port, service in [(11434, "Ollama"), (6333, "Qdrant"), (6379, "Redis"), (8000, "Gateway")]:
        if _check_port(port):
            console.print(f"  [green]PASS[/] Port {port} ({service}) is available")
        else:
            console.print(f"  [yellow]WARN[/] Port {port} ({service}) is in use")

    # Config
    try:
        from llmstack.config.loader import load_config
        load_config()
        console.print("  [green]PASS[/] llmstack.yaml is valid")
    except FileNotFoundError:
        console.print("  [yellow]WARN[/] No llmstack.yaml found (run 'llmstack init')")
    except SystemExit:
        console.print("  [red]FAIL[/] llmstack.yaml has validation errors")
        issues += 1

    if issues:
        console.print(f"\n[error]{issues} issue(s) found.[/]")
    else:
        console.print(f"\n[success]All checks passed![/]")
