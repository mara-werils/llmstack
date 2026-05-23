"""llmstack status — show health of all services."""

from __future__ import annotations

import httpx
from rich.table import Table

from llmstack.cli.console import console, banner, success, failure, warn
from llmstack.docker.manager import DockerManager


def status() -> None:
    """Show the status of all running llmstack services."""
    banner("LLMStack Status")

    docker = DockerManager()
    services = docker.list_services()

    if not services:
        console.print("\n[muted]No llmstack services are running.[/]")
        console.print("[muted]Run [bold]llmstack up[/] to start.\n")
        return

    table = Table(show_header=True, show_edge=False)
    table.add_column("Service", style="cyan")
    table.add_column("Container", style="muted")
    table.add_column("Status")
    table.add_column("Ports")
    table.add_column("Uptime", style="muted")

    running = 0
    for svc in services:
        is_running = svc["status"] == "running"
        if is_running:
            running += 1
        status_style = "green" if is_running else "red"
        ports_str = ""
        if svc["ports"]:
            port_list = []
            for container_port, host_bindings in svc["ports"].items():
                if host_bindings:
                    for binding in host_bindings:
                        port_list.append(f"{binding['HostPort']}->{container_port}")
            ports_str = ", ".join(port_list)

        # Try to get uptime from container
        uptime = svc.get("uptime", "")

        table.add_row(
            svc["name"],
            svc["container_id"][:12],
            f"[{status_style}]{svc['status']}[/]",
            ports_str,
            uptime,
        )

    console.print(table)
    console.print(f"\n  [accent]{running}/{len(services)}[/] services running")

    # Quick gateway health check
    try:
        from llmstack.config.loader import load_config
        config = load_config()
        gw_url = f"http://localhost:{config.gateway.port}"
        resp = httpx.get(f"{gw_url}/healthz", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            gw_status = data.get("status", "unknown")
            if gw_status == "ok":
                success(f"Gateway healthy at {gw_url}")
            else:
                warn(f"Gateway degraded at {gw_url}")
        else:
            failure(f"Gateway returned {resp.status_code}")
    except Exception:
        pass

    console.print()
