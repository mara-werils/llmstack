"""llmstack plugin — Discover and manage plugins."""

from __future__ import annotations

from llmstack.cli.console import console


def plugin_list() -> None:
    """List all discovered plugins."""
    from rich.table import Table
    from llmstack.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    plugins = registry.discover()

    if not plugins:
        console.print("[dim]No plugins installed.[/]")
        console.print("[dim]Install plugins with: pip install llmstack-plugin-name[/]")
        return

    table = Table(
        title="Installed Plugins",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Type", width=12)
    table.add_column("Status", width=10)
    table.add_column("Description")

    for p in plugins:
        status = "[green]enabled[/]" if p.enabled else "[red]disabled[/]"
        table.add_row(p.name, p.version, p.plugin_type, status, p.description)

    console.print(table)


def plugin_enable(name: str) -> None:
    """Enable a plugin."""
    from llmstack.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    if registry.enable(name):
        console.print(f"[green]Plugin '{name}' enabled.[/]")
    else:
        console.print(f"[error]Plugin not found: {name}[/]")


def plugin_disable(name: str) -> None:
    """Disable a plugin."""
    from llmstack.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    if registry.disable(name):
        console.print(f"[yellow]Plugin '{name}' disabled.[/]")
    else:
        console.print(f"[error]Plugin not found: {name}[/]")


def plugin_info(name: str) -> None:
    """Show plugin details."""
    from rich.panel import Panel
    from llmstack.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    for p in registry.discover():
        if p.name == name:
            console.print(
                Panel(
                    f"[bold]Name:[/] {p.name}\n"
                    f"[bold]Version:[/] {p.version}\n"
                    f"[bold]Type:[/] {p.plugin_type}\n"
                    f"[bold]Author:[/] {p.author or 'unknown'}\n"
                    f"[bold]Entry Point:[/] {p.entry_point}\n"
                    f"[bold]Enabled:[/] {'yes' if p.enabled else 'no'}\n"
                    f"[bold]Description:[/] {p.description}",
                    title=f"Plugin: {name}",
                    border_style="cyan",
                )
            )
            return

    console.print(f"[error]Plugin not found: {name}[/]")
