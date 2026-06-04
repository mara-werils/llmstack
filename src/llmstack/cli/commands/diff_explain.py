"""llmstack diff — explain git diffs in plain English."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from llmstack.cli.console import console


DIFF_SYSTEM_PROMPT = """You are a senior engineer reviewing git changes. Explain this diff clearly.

Structure your response as:
1. **What changed** (1-2 sentences)
2. **Why it likely changed** (infer from context)
3. **Impact** (what this affects: behavior, performance, API, etc.)
4. **Risk level**: LOW / MEDIUM / HIGH (with brief reason)

Be concise and technical. Use bullet points for multiple changes."""


def diff_explain(
    target: str = "HEAD~1..HEAD",
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    staged: bool = False,
    commits: int = 1,
    file: str | None = None,
) -> None:
    """Explain git diff in plain English."""
    asyncio.run(
        _diff_async(
            target=target,
            model=model,
            ollama_url=ollama_url,
            staged=staged,
            commits=commits,
            file=file,
        )
    )


def _get_diff(target: str, staged: bool, commits: int, file: str | None) -> str:
    """Get git diff."""
    cwd = str(Path.cwd())

    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    extra = ["--", file] if file else []

    if staged:
        return run_git("diff", "--staged", *extra)
    elif ".." in target:
        return run_git("diff", target, *extra)
    else:
        return run_git("diff", f"HEAD~{commits}..HEAD", *extra)


async def _diff_async(
    target: str,
    model: str,
    ollama_url: str,
    staged: bool,
    commits: int,
    file: str | None,
) -> None:
    import httpx
    import typer
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

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

    diff = _get_diff(target, staged, commits, file)
    if not diff:
        console.print("[warning]No diff found.[/]")
        raise typer.Exit(0)

    # Get git log for context
    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
                timeout=10,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    log = run_git("log", "--oneline", f"-{commits + 1}")

    MAX_DIFF = 8000
    if len(diff) > MAX_DIFF:
        diff = diff[:MAX_DIFF] + "\n\n... (truncated)"

    console.print()
    console.print(f"[bold]llmstack diff[/]  model=[cyan]{model}[/]")
    console.print()

    prompt = f"""Explain this git diff:

{f"Recent commits:{chr(10)}{log}{chr(10)}" if log else ""}
Diff:
{diff}"""

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]Analyzing diff..."), console=console
    ) as progress:
        task = progress.add_task("Analyzing", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        result = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": DIFF_SYSTEM_PROMPT},
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

    console.print()
    console.print(Panel(Markdown(result), title="Diff Explanation", border_style="cyan"))
    console.print()
