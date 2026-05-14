"""llmstack watch — real-time file analysis with AI suggestions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import datetime

from llmstack.cli.console import console


WATCH_SYSTEM_PROMPT = """You are a real-time code assistant. A file was just saved.
Quickly analyze the recent changes and provide concise, actionable suggestions.

Focus on:
- Immediate bugs or errors
- Missing error handling
- Performance anti-patterns
- Security issues

Be VERY concise - max 3-4 bullet points. Skip praise."""


def watch(
    directory: str = ".",
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    patterns: str = "*.py,*.js,*.ts",
    debounce: float = 2.0,
) -> None:
    """Watch files for changes and get real-time AI suggestions."""
    asyncio.run(_watch_async(
        directory=directory, model=model, ollama_url=ollama_url,
        patterns=patterns, debounce=debounce,
    ))


async def _watch_async(
    directory: str,
    model: str,
    ollama_url: str,
    patterns: str,
    debounce: float,
) -> None:
    import typer
    from rich.panel import Panel
    from rich.markdown import Markdown

    ollama_url = ollama_url.rstrip("/")
    watch_dir = Path(directory).resolve()

    console.print()
    console.print(Panel(
        f"[bold]Watching:[/] {watch_dir}\n"
        f"[bold]Patterns:[/] {patterns}\n"
        f"[bold]Model:[/] {model}\n\n"
        f"[dim]Save a file to get AI suggestions. Ctrl+C to stop.[/]",
        title="llmstack watch", border_style="cyan",
    ))
    console.print()

    pat_list = [p.strip() for p in patterns.split(",")]
    last_analyzed: dict[str, float] = {}
    mtimes: dict[str, float] = {}

    # Initial scan
    for pat in pat_list:
        for fpath in watch_dir.rglob(pat):
            if not any(x in str(fpath) for x in ["__pycache__", ".git", "node_modules"]):
                try:
                    mtimes[str(fpath)] = fpath.stat().st_mtime
                except OSError:
                    pass

    async def analyze_file(fpath: Path) -> None:
        """Analyze a changed file."""
        now = asyncio.get_event_loop().time()
        key = str(fpath)
        if now - last_analyzed.get(key, 0) < debounce:
            return
        last_analyzed[key] = now

        try:
            content = fpath.read_text(errors="replace")
            if len(content) > 6000:
                content = content[:6000] + "\n... (truncated)"
        except OSError:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        console.print(f"[dim]{timestamp}[/] [bold cyan]Changed:[/] {fpath.relative_to(watch_dir)}")

        import httpx
        timeout = httpx.Timeout(60, connect=5, read=60, write=10)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": WATCH_SYSTEM_PROMPT},
                            {"role": "user", "content": f"File: {fpath.name}\n\n```\n{content}\n```"},
                        ],
                        "stream": True,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        return
                    result = ""
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

                if result.strip():
                    console.print(Panel(
                        Markdown(result),
                        title=f"Suggestions for {fpath.name}",
                        border_style="green",
                    ))
        except Exception:
            pass

    # Watch loop
    try:
        while True:
            await asyncio.sleep(1.0)
            for pat in pat_list:
                for fpath in watch_dir.rglob(pat):
                    if any(x in str(fpath) for x in ["__pycache__", ".git", "node_modules"]):
                        continue
                    key = str(fpath)
                    try:
                        mtime = fpath.stat().st_mtime
                        if key in mtimes and mtime > mtimes[key]:
                            mtimes[key] = mtime
                            asyncio.ensure_future(analyze_file(fpath))
                        elif key not in mtimes:
                            # Newly created file detected
                            mtimes[key] = mtime
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            console.print(
                                f"[dim]{timestamp}[/] [bold green]New file:[/] "
                                f"{fpath.relative_to(watch_dir)}"
                            )
                            asyncio.ensure_future(analyze_file(fpath))
                    except OSError:
                        pass
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/]")
