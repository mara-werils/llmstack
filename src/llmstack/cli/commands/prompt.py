"""llmstack prompt — Manage and use prompt templates."""

from __future__ import annotations

from llmstack.cli.console import console


def prompt_list(category: str | None = None) -> None:
    """List all available prompt templates."""
    from rich.table import Table
    from llmstack.prompts.templates import TemplateManager

    mgr = TemplateManager()
    templates = mgr.list_all(category=category)

    if not templates:
        console.print("[dim]No templates found.[/]")
        return

    table = Table(
        title="Prompt Templates",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold")
    table.add_column("Category", width=14)
    table.add_column("Description")
    table.add_column("Variables")
    table.add_column("Type", width=8)

    for t in templates:
        table.add_row(
            t.name,
            t.category,
            t.description,
            ", ".join(t.variables),
            "[dim]builtin[/]" if t.is_builtin else "[green]custom[/]",
        )

    console.print(table)

    cats = mgr.categories()
    console.print(f"\n[dim]Categories: {', '.join(cats)}[/]")


def prompt_show(name: str) -> None:
    """Show a specific template."""
    from rich.panel import Panel
    from llmstack.prompts.templates import TemplateManager

    mgr = TemplateManager()
    template = mgr.get(name)

    if not template:
        console.print(f"[error]Template not found: {name}[/]")
        return

    console.print()
    console.print(f"[bold]{template.name}[/]  category=[dim]{template.category}[/]")
    console.print(f"  {template.description}")
    console.print(f"  Variables: [cyan]{', '.join(template.variables)}[/]")
    console.print()
    console.print(Panel(template.template, title="Template", border_style="cyan"))
    console.print()
    console.print("[dim]Usage: llmstack prompt use <name> --var key=value[/]")


def prompt_use(name: str, variables: list[str] | None = None) -> None:
    """Render a template with variables."""
    from rich.panel import Panel
    from llmstack.prompts.templates import TemplateManager

    mgr = TemplateManager()
    template = mgr.get(name)

    if not template:
        console.print(f"[error]Template not found: {name}[/]")
        return

    # Parse key=value pairs
    var_dict = {}
    for v in variables or []:
        if "=" in v:
            key, val = v.split("=", 1)
            var_dict[key.strip()] = val.strip()

    try:
        rendered = mgr.render(name, **var_dict)
        console.print()
        console.print(Panel(rendered, title=f"Rendered: {name}", border_style="green"))
    except ValueError as e:
        console.print(f"[error]{e}[/]")


def prompt_create(
    name: str,
    template: str,
    description: str = "",
    category: str = "custom",
) -> None:
    """Create a custom prompt template."""
    from llmstack.prompts.templates import TemplateManager

    mgr = TemplateManager()
    result = mgr.save(name=name, template=template, description=description, category=category)
    console.print(f"[green]Template saved:[/] [bold]{result.name}[/]")
    console.print(f"  Variables: {', '.join(result.variables)}")


def prompt_delete(name: str) -> None:
    """Delete a custom template."""
    from llmstack.prompts.templates import TemplateManager

    mgr = TemplateManager()
    if mgr.delete(name):
        console.print(f"[green]Template '{name}' deleted.[/]")
    else:
        console.print(f"[error]Template not found or is a builtin: {name}[/]")
