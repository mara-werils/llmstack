"""llmstack scaffold — Generate project structure from description using AI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


SCAFFOLD_SYSTEM_PROMPT = """You are an expert software architect who creates production-ready project scaffolds.

Given a project description, generate a complete project structure with file contents.

Output a JSON array of files to create:
[
  {"path": "src/main.py", "content": "file content here"},
  {"path": "README.md", "content": "# Project Name\\n..."},
  {"path": "requirements.txt", "content": "fastapi>=0.100\\n..."}
]

Rules:
- Include all necessary config files (pyproject.toml/package.json, .gitignore, etc.)
- Include a README with setup instructions
- Include basic tests
- Use modern best practices for the chosen tech stack
- Include type annotations
- Include proper .gitignore
- Don't include lock files or generated code

Output ONLY the JSON array, no explanations."""


PROJECT_PRESETS = {
    "fastapi": "Python FastAPI REST API with SQLAlchemy, Pydantic models, async endpoints, and pytest tests",
    "flask": "Python Flask web application with Blueprints, SQLAlchemy, and templates",
    "django": "Django project with REST framework, models, serializers, and views",
    "nextjs": "Next.js 14 App Router project with TypeScript, Tailwind CSS, and API routes",
    "react": "React + TypeScript + Vite project with components, hooks, and tests",
    "cli-python": "Python CLI application with Typer, Rich output, and config management",
    "cli-go": "Go CLI application with Cobra, structured logging, and tests",
    "express": "Node.js Express API with TypeScript, middleware, and Jest tests",
    "rust-cli": "Rust CLI application with clap, error handling, and tests",
    "chrome-ext": "Chrome extension with manifest v3, popup, content script, and background worker",
    "discord-bot": "Discord bot with slash commands, event handlers, and database",
    "telegram-bot": "Telegram bot with commands, inline keyboards, and state management",
}


def scaffold(
    description: str,
    preset: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output_dir: str = ".",
    dry_run: bool = False,
) -> None:
    """Generate project structure from description."""
    asyncio.run(_scaffold_async(
        description=description, preset=preset, model=model,
        ollama_url=ollama_url, output_dir=output_dir, dry_run=dry_run,
    ))


async def _scaffold_async(
    description: str,
    preset: str | None,
    model: str,
    ollama_url: str,
    output_dir: str,
    dry_run: bool,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.tree import Tree

    ollama_url = ollama_url.rstrip("/")

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print("[error]Ollama is not responding.[/]")
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(Panel("[error]Cannot connect to Ollama.[/]", border_style="red"))
        raise typer.Exit(1)

    # Use preset if provided
    if preset:
        if preset in PROJECT_PRESETS:
            description = f"{PROJECT_PRESETS[preset]}. {description}" if description else PROJECT_PRESETS[preset]
        else:
            console.print(f"[error]Unknown preset: {preset}[/]")
            console.print(f"Available: {', '.join(PROJECT_PRESETS.keys())}")
            raise typer.Exit(1)

    if not description:
        console.print("[error]Provide a project description or --preset[/]")
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold]llmstack scaffold[/]  model=[cyan]{model}[/]")
    if preset:
        console.print(f"  [dim]Preset: {preset}[/]")
    console.print(f"  [dim]{description[:100]}{'...' if len(description) > 100 else ''}[/]")
    console.print()

    prompt = f"""Generate a complete project scaffold for this description:

{description}

Output a JSON array of files with their full contents.
Each file: {{"path": "relative/path", "content": "file content"}}

Include:
- Main source files
- Configuration (pyproject.toml or package.json, .gitignore)
- README.md with setup instructions
- Basic test files
- Type annotations and docstrings

Output ONLY valid JSON array, no markdown fences or explanations."""

    # Stream response
    full_response = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating project scaffold..."),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SCAFFOLD_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    # Parse JSON response
    # Strip code fences
    cleaned = full_response.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    # Find JSON array
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        cleaned = cleaned[bracket_start:bracket_end + 1]

    try:
        files = json.loads(cleaned)
    except json.JSONDecodeError:
        console.print("[error]Failed to parse scaffold response as JSON.[/]")
        console.print(Panel(full_response[:2000], title="Raw Response", border_style="red"))
        return

    if not isinstance(files, list):
        console.print("[error]Expected a JSON array of files.[/]")
        return

    # Display file tree
    tree = Tree(f"[bold]{output_dir}[/]")
    for f in sorted(files, key=lambda x: x.get("path", "")):
        path = f.get("path", "")
        content = f.get("content", "")
        size = len(content)
        tree.add(f"[cyan]{path}[/]  [dim]({size} chars)[/]")

    console.print()
    console.print(tree)
    console.print(f"\n[bold]{len(files)} files[/] to create")

    if dry_run:
        console.print("\n[dim]Dry run — no files created.[/]")
        return

    # Create files
    out_dir = Path(output_dir)
    created = 0
    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")
        if not path:
            continue

        file_path = out_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            console.print(f"  [yellow]SKIP[/] {path} (already exists)")
            continue

        file_path.write_text(content)
        created += 1

    console.print(f"\n[green]{created} files created in {output_dir}[/]")
