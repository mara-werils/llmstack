"""llmstack fix — AI-powered auto-fix for code issues."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


FIX_SYSTEM_PROMPT = """You are an expert software engineer. Your task is to fix the specific issue in the code.

Output ONLY a unified diff patch that can be applied with `patch -p0`.
Start with --- and +++ lines.
Be minimal - only change what is necessary to fix the issue.
Do not add comments or explanations in the output - just the patch."""


def fix(
    description: str = "",
    file: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    dry_run: bool = False,
    interactive: bool = True,
) -> None:
    """Auto-fix code issues with AI."""
    asyncio.run(_fix_async(
        description=description, file=file, model=model,
        ollama_url=ollama_url, dry_run=dry_run, interactive=interactive,
    ))


async def _fix_async(
    description: str,
    file: str | None,
    model: str,
    ollama_url: str,
    dry_run: bool,
    interactive: bool,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

    ollama_url = ollama_url.rstrip("/")

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print("[error]Ollama is not responding.[/]")
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(Panel(
            "[error]Cannot connect to Ollama.[/]",
            title="Connection Error", border_style="red",
        ))
        raise typer.Exit(1)

    # Read target file
    file_content = ""
    file_path = None
    if file:
        file_path = Path(file)
        if file_path.exists():
            file_content = file_path.read_text(errors="replace")

    if not description and not file_content:
        console.print("[error]Provide --description of the issue or --file to fix.[/]")
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold]llmstack fix[/]  model=[cyan]{model}[/]")
    if file_path:
        console.print(f"  [dim]File: {file_path}[/]")
    console.print()

    prompt_parts = []
    if description:
        prompt_parts.append(f"Issue to fix: {description}")
    if file_content:
        prompt_parts.append(f"File: {file_path}\n\n```\n{file_content[:8000]}\n```")
    prompt_parts.append("Generate a minimal unified diff patch to fix this issue.")

    prompt = "\n\n".join(prompt_parts)

    # Get fix from LLM
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating fix..."),
        console=console,
    ) as progress:
        task = progress.add_task("Fixing", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        patch = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": FIX_SYSTEM_PROMPT},
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
                            patch += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    # Extract patch from code block if wrapped
    if "```diff" in patch:
        patch = patch.split("```diff", 1)[1].split("```", 1)[0].strip()
    elif "```" in patch:
        patch = patch.split("```", 1)[1].split("```", 1)[0].strip()

    if not patch.strip() or not patch.strip().startswith("---"):
        console.print()
        console.print(Panel(patch, title="Suggested Fix", border_style="cyan"))
        console.print("[dim]Note: Could not generate a patchable diff. Review suggestion above.[/]")
        return

    # Show patch
    console.print()
    console.print(Syntax(patch, "diff", theme="monokai", line_numbers=True))
    console.print()

    if dry_run:
        console.print("[dim]Dry run — patch not applied.[/]")
        return

    # Apply patch interactively
    if interactive:
        try:
            confirm = console.input("[bold]Apply this patch? [y/N][/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/]")
            return

        if confirm != "y":
            console.print("[dim]Patch not applied.[/]")
            return

    # Write patch to temp file and apply
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        result = subprocess.run(
            ["patch", "-p0", "--input", patch_file],
            capture_output=True, text=True, cwd=str(Path.cwd()),
        )
        if result.returncode == 0:
            console.print("[green]Patch applied successfully.[/]")
        else:
            console.print(f"[error]Patch failed: {result.stderr}[/]")
    except FileNotFoundError:
        console.print("[error]`patch` command not found. Apply the diff manually.[/]")
    finally:
        Path(patch_file).unlink(missing_ok=True)
