"""llmstack commit — AI-powered commit message generation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from llmstack.cli.console import console


COMMIT_SYSTEM_PROMPT = """You are an expert at writing clear, conventional git commit messages.

Format: <type>(<scope>): <short description>

Types: feat, fix, docs, style, refactor, test, chore, perf, ci
- Keep the subject line under 72 characters
- Use imperative mood ("add" not "added")
- Be specific and meaningful

Output ONLY the commit message (subject line, optionally body after blank line).
No preamble, no explanation."""


def commit_gen(
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    staged: bool = True,
    push: bool = False,
    all_changes: bool = False,
    message: str | None = None,
) -> None:
    """Generate and optionally apply a commit message."""
    asyncio.run(_commit_async(
        model=model, ollama_url=ollama_url, staged=staged,
        push=push, all_changes=all_changes, message=message,
    ))


async def _commit_async(
    model: str,
    ollama_url: str,
    staged: bool,
    push: bool,
    all_changes: bool,
    message: str | None,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    ollama_url = ollama_url.rstrip("/")
    cwd = str(Path.cwd())

    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=30,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    # Stage all if requested
    if all_changes:
        subprocess.run(["git", "add", "-A"], cwd=cwd)

    # Get staged diff
    diff = run_git("diff", "--staged")
    if not diff:
        console.print("[warning]No staged changes found.[/]")
        console.print("[dim]Tip: Stage files with 'git add' first, or use --all to stage everything.[/]")
        raise typer.Exit(0)

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

    MAX_DIFF = 6000
    if len(diff) > MAX_DIFF:
        diff = diff[:MAX_DIFF] + "\n... (truncated)"

    console.print()
    console.print(f"[bold]llmstack commit[/]  model=[cyan]{model}[/]")
    console.print()

    prompt = f"""Generate a commit message for this diff:

{diff}"""

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Generating commit message..."), console=console) as progress:
        task = progress.add_task("Generating", total=None)

        timeout = httpx.Timeout(120, connect=10, read=120, write=30)
        result = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": COMMIT_SYSTEM_PROMPT},
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
                            result += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    commit_msg = result.strip()

    console.print()
    console.print(Panel(
        f"[bold green]{commit_msg}[/]",
        title="Suggested Commit Message",
        border_style="cyan",
    ))
    console.print()

    # Interactive confirmation
    try:
        action = console.input(
            "[bold]Apply? [[bold green]y[/]/[bold yellow]e[/]dit/[bold red]n[/]o][/] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/]")
        return

    if action == "e":
        try:
            edited = console.input("[bold]Edit message:[/] ").strip()
            if edited:
                commit_msg = edited
        except (EOFError, KeyboardInterrupt):
            return

    if action in ("y", "e", ""):
        git_result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True, text=True, cwd=cwd,
        )
        if git_result.returncode == 0:
            console.print("[green]Committed.[/]")
            if push:
                push_result = subprocess.run(
                    ["git", "push"], capture_output=True, text=True, cwd=cwd,
                )
                if push_result.returncode == 0:
                    console.print("[green]Pushed.[/]")
                else:
                    console.print(f"[error]Push failed: {push_result.stderr}[/]")
        else:
            console.print(f"[error]Commit failed: {git_result.stderr}[/]")
    else:
        console.print("[dim]Commit cancelled.[/]")
