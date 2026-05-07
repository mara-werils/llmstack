"""llmstack status — show health of all services."""

from __future__ import annotations

from rich.table import Table

from llmstack.cli.console import console
from llmstack.docker.manager import DockerManager


def status() -> None:
    """Show the status of all running llmstack services."""
    docker = DockerManager()
    services = docker.list_services()

    if not services:
        console.print("[info]No llmstack services are running.[/]")
        console.print("Run [bold]llmstack up[/] to start.")
        return

    table = Table(title="LLMStack Status", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Container")
    table.add_column("Status")
    table.add_column("Ports")

    for svc in services:
        status_style = "green" if svc["status"] == "running" else "red"
        ports_str = ""
        if svc["ports"]:
            port_list = []
            for container_port, host_bindings in svc["ports"].items():
                if host_bindings:
                    for binding in host_bindings:
                        port_list.append(f"{binding['HostPort']}->{container_port}")
            ports_str = ", ".join(port_list)

        table.add_row(
            svc["name"],
            svc["container_id"],
            f"[{status_style}]{svc['status']}[/]",
            ports_str,
        )

    console.print(table)
