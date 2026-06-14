"""Pre-flight checks for booting the stack: port availability and Docker daemon.

Single source of truth shared by ``llmstack up`` (a hard gate that aborts early
with an actionable message) and ``llmstack doctor`` (a diagnostic). A taken port
should read as "Port 6379 (Redis) is in use by redis-server" — not an opaque
Docker bind error after half the stack has already started.
"""

from __future__ import annotations

import shutil
import socket
from dataclasses import dataclass


@dataclass
class PortCheck:
    """Availability of a single host port the stack wants to bind."""

    port: int
    service: str
    available: bool
    owner: str | None = None  # best-effort name of the process holding the port


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if ``port`` is free to bind on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) != 0


def port_owner(port: int) -> str | None:
    """Best-effort name of the process listening on ``port`` (None if unknown).

    Returns None when psutil is unavailable or the OS denies the lookup (common
    for ports owned by other users on macOS) — callers must treat None as
    "an unknown process".
    """
    try:
        import psutil
    except Exception:
        return None

    try:
        for conn in psutil.net_connections(kind="inet"):
            laddr = conn.laddr
            if laddr and laddr.port == port and conn.status == psutil.CONN_LISTEN:
                if not conn.pid:
                    return None
                try:
                    return psutil.Process(conn.pid).name()
                except Exception:
                    return None
    except Exception:
        # AccessDenied / platform quirks — degrade gracefully.
        return None
    return None


def check_ports(ports: list[tuple[int, str]]) -> list[PortCheck]:
    """Resolve availability (and owner, when taken) for each ``(port, service)``."""
    results: list[PortCheck] = []
    for port, service in ports:
        available = is_port_available(port)
        owner = None if available else port_owner(port)
        results.append(PortCheck(port=port, service=service, available=available, owner=owner))
    return results


def required_ports(config) -> list[tuple[int, str]]:
    """Host ports the stack will bind for ``config``, derived from the schema.

    Excludes the inference port (Ollama 11434): an already-running Ollama on the
    host is the expected case, not a conflict.
    """
    ports: list[tuple[int, str]] = [
        (config.services.vectors.port, "Qdrant"),
        (config.services.cache.port, "Redis"),
        (config.gateway.port, "Gateway"),
    ]
    if getattr(config.observe, "metrics", False):
        ports.append((9090, "Prometheus"))
        ports.append((config.observe.dashboard_port, "Grafana"))
    return ports


def docker_status() -> str | None:
    """Return None if Docker is usable, else an actionable error string."""
    if not shutil.which("docker"):
        return (
            "Docker is not installed. Install it from "
            "https://docs.docker.com/get-docker/ then re-run [bold cyan]llmstack up[/]."
        )
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return None
    except Exception:
        return (
            "Docker is installed but its daemon isn't reachable. "
            "Start Docker Desktop (or [bold cyan]sudo systemctl start docker[/]) and retry."
        )
