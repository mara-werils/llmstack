"""llmstack changelog — Auto-generate changelogs from git history using AI."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from llmstack.cli.console import console


CHANGELOG_SYSTEM_PROMPT = """You are a technical writer creating a changelog from git commits.

Generate a well-structured changelog in Keep a Changelog format:
- Group changes by type: Added, Changed, Deprecated, Removed, Fixed, Security
- Write user-facing descriptions (not commit messages verbatim)
- Merge related commits into single entries
- Skip trivial changes (typo fixes, formatting)
- Use present tense, imperative mood
- Include breaking changes prominently

Output in Markdown format starting with ## [version] - date"""


def changelog(
    since: str | None = None,
    version: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output: str | None = None,
    max_commits: int = 100,
) -> None:
    """Generate a changelog from git history."""
    asyncio.run(_changelog_async(
        since=since, version=version, model=model,
        ollama_url=ollama_url, output=output, max_commits=max_commits,
    ))


def _get_git_log(since: str | None, max_commits: int) -> str:
    """Get git log for changelog generation."""
    cmd = ["git", "log", f"--max-count={max_commits}", "--format=%H|%s|%an|%ai"]
    if since:
        cmd.append(f"{since}..HEAD")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def _get_tags() -> list[str]:
    """Get recent git tags."""
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-creatordate", "-l"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip().split("\n")[:5] if result.returncode == 0 else []
    except Exception:
        return []


def _get_diff_stats(since: str | None) -> str:
    """Get diff stats summary."""
    cmd = ["git", "diff", "--stat"]
    if since:
        cmd.append(f"{since}..HEAD")
    else:
        cmd.extend(["HEAD~20..HEAD"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


async def _changelog_async(
    since: str | None,
    version: str | None,
    model: str,
    ollama_url: str,
    output: str | None,
    max_commits: int,
) -> None:
    import httpx
    import typer
    from datetime import date
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown

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

    git_log = _get_git_log(since, max_commits)
    if not git_log:
        console.print("[warning]No git commits found.[/]")
        raise typer.Exit(0)

    tags = _get_tags()
    diff_stats = _get_diff_stats(since)
    version_str = version or "Unreleased"
    today = date.today().isoformat()

    commit_count = len(git_log.strip().split("\n"))
    console.print()
    console.print(f"[bold]llmstack changelog[/]  model=[cyan]{model}[/]  commits=[dim]{commit_count}[/]")
    if since:
        console.print(f"  [dim]Since: {since}[/]")
    console.print()

    prompt = f"""Generate a changelog from these git commits.

Version: {version_str}
Date: {today}
Recent tags: {', '.join(tags) if tags else 'none'}

Git log (hash|subject|author|date):
{git_log[:10000]}

Diff stats:
{diff_stats[:3000]}

Generate a well-formatted Markdown changelog following Keep a Changelog conventions."""

    # Stream LLM response
    changelog_text = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating changelog..."),
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
                        {"role": "system", "content": CHANGELOG_SYSTEM_PROMPT},
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
                            changelog_text += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    console.print()
    console.print(Markdown(changelog_text))

    if output:
        Path(output).write_text(changelog_text + "\n")
        console.print(f"\n[green]Changelog saved to {output}[/]")
