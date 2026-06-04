"""llmstack workflow — Run automated command pipelines."""

from __future__ import annotations

from llmstack.cli.console import console


def workflow_list() -> None:
    """List all available workflows."""
    from rich.table import Table
    from llmstack.workflows.engine import WorkflowEngine

    engine = WorkflowEngine()
    workflows = engine.list_workflows()

    if not workflows:
        console.print("[dim]No workflows available.[/]")
        return

    table = Table(
        title="Available Workflows",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold")
    table.add_column("Title")
    table.add_column("Description")
    table.add_column("Steps", justify="right", width=5)
    table.add_column("Type", width=8)

    for wf in workflows:
        wf_type = "[dim]builtin[/]" if wf["builtin"] else "[green]custom[/]"
        table.add_row(
            wf["name"],
            wf["title"],
            wf["description"],
            str(wf["steps"]),
            wf_type,
        )

    console.print(table)
    console.print("\n[dim]Run: llmstack workflow run <name>[/]")


def workflow_show(name: str) -> None:
    """Show workflow details."""
    from rich.panel import Panel
    from rich.tree import Tree
    from llmstack.workflows.engine import WorkflowEngine

    engine = WorkflowEngine()
    wf = engine.get_workflow(name)

    if not wf:
        console.print(f"[error]Workflow not found: {name}[/]")
        return

    tree = Tree(f"[bold]{wf['name']}[/]")
    for i, step in enumerate(wf["steps"], 1):
        args_str = " ".join(f"--{k}={v}" for k, v in step.get("args", {}).items() if v)
        tree.add(f"[cyan]{i}.[/] {step['command']} {args_str}")

    console.print()
    console.print(
        Panel(
            f"{wf.get('description', '')}\n\n",
            title=f"Workflow: {name}",
            border_style="cyan",
        )
    )
    console.print(tree)


def workflow_run(name: str) -> None:
    """Run a workflow pipeline."""
    from rich.panel import Panel
    from llmstack.workflows.engine import WorkflowEngine

    engine = WorkflowEngine()
    wf = engine.get_workflow(name)

    if not wf:
        console.print(f"[error]Workflow not found: {name}[/]")
        return

    console.print()
    console.print(
        Panel(
            f"[bold]{wf['name']}[/]\n{wf.get('description', '')}\n\nSteps: {len(wf['steps'])}",
            title=f"Running Workflow: {name}",
            border_style="cyan",
        )
    )

    import time

    start = time.time()
    completed = 0
    failed = 0

    for i, step in enumerate(wf["steps"], 1):
        console.print()
        console.print(f"[bold cyan]━━━ Step {i}/{len(wf['steps'])}: {step['name']} ━━━[/]")
        console.print()

        try:
            _run_step(step)
            completed += 1
        except Exception as e:
            console.print(f"[error]Step failed: {e}[/]")
            failed += 1
            if not step.get("continue_on_error", True):
                console.print("[error]Workflow aborted.[/]")
                break

    duration = time.time() - start
    status_color = "green" if failed == 0 else "yellow"

    console.print()
    console.print(
        Panel(
            f"[bold]Completed:[/] {completed}/{len(wf['steps'])}\n"
            f"[bold]Failed:[/] {failed}\n"
            f"[bold]Duration:[/] {duration:.1f}s",
            title="Workflow Complete",
            border_style=status_color,
        )
    )


def _run_step(step: dict) -> None:
    """Run a single workflow step by invoking the corresponding CLI command."""
    command = step["command"]
    args = step.get("args", {})

    # Map commands to their implementation functions
    command_map = {
        "complexity": lambda: _import_and_run(
            "llmstack.cli.commands.complexity", "complexity", args
        ),
        "dead-code": lambda: _import_and_run("llmstack.cli.commands.dead_code", "dead_code", args),
        "security": lambda: _import_and_run("llmstack.cli.commands.security", "security", args),
        "review": lambda: _import_and_run("llmstack.cli.commands.review", "review", args),
        "tokens": lambda: _import_and_run("llmstack.cli.commands.tokens", "tokens", args),
        "deps": lambda: _import_and_run("llmstack.cli.commands.deps", "deps", args),
        "diagram": lambda: _import_and_run("llmstack.cli.commands.diagram", "diagram", args),
        "changelog": lambda: _import_and_run("llmstack.cli.commands.changelog", "changelog", args),
        "info": lambda: _import_and_run("llmstack.cli.commands.info", "info", {}),
        "commit": lambda: _import_and_run("llmstack.cli.commands.commit_gen", "commit_gen", args),
        "explain": lambda: _import_and_run("llmstack.cli.commands.explain", "explain", args),
        "refactor": lambda: _import_and_run("llmstack.cli.commands.refactor", "refactor", args),
    }

    handler = command_map.get(command)
    if handler:
        handler()
    else:
        console.print(f"[warning]Unknown command: {command}, skipping[/]")


def _import_and_run(module_path: str, func_name: str, args: dict) -> None:
    """Dynamically import and run a command function."""
    import importlib

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    func(**args)


def workflow_create(name: str, steps_json: str, description: str = "") -> None:
    """Create a custom workflow from JSON steps."""
    import json
    from llmstack.workflows.engine import WorkflowEngine

    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError:
        console.print("[error]Invalid JSON for steps.[/]")
        return

    engine = WorkflowEngine()
    workflow = {
        "name": name,
        "description": description,
        "steps": steps,
    }
    engine.save_custom(name, workflow)
    console.print(f"[green]Workflow '{name}' saved with {len(steps)} steps.[/]")


def workflow_delete(name: str) -> None:
    """Delete a custom workflow."""
    from llmstack.workflows.engine import WorkflowEngine

    engine = WorkflowEngine()
    if engine.delete_custom(name):
        console.print(f"[green]Workflow '{name}' deleted.[/]")
    else:
        console.print(f"[error]Custom workflow not found: {name}[/]")
