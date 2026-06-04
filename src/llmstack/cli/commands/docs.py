"""llmstack docs — AI-powered documentation generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


DOCS_SYSTEM_PROMPT = """You are a technical writer and expert programmer. Generate clear, concise documentation.

When generating docstrings:
- Use the appropriate format for the language (Python: Google style, JS: JSDoc)
- Be concise but complete
- Include Args, Returns, Raises sections for Python
- Document parameters and return type for JS/TS

When generating README:
- Lead with a one-line description
- Show the most common usage immediately
- Include installation, quickstart, and API reference sections
- Use code blocks for examples

Output ONLY the documentation content, no preamble."""


def docs(
    target: str | None = None,
    output: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    doc_type: str = "docstrings",
    write: bool = False,
) -> None:
    """Generate documentation for code."""
    asyncio.run(
        _docs_async(
            target=target,
            output=output,
            model=model,
            ollama_url=ollama_url,
            doc_type=doc_type,
            write=write,
        )
    )


async def _docs_async(
    target: str | None,
    output: str | None,
    model: str,
    ollama_url: str,
    doc_type: str,
    write: bool,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel

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

    # Determine target
    target_path = Path(target) if target else Path.cwd()

    if doc_type == "readme":
        await _generate_readme(target_path, model, ollama_url, output, write)
    else:
        await _generate_docstrings(target_path, model, ollama_url, output, write)


async def _generate_readme(
    target_path: Path,
    model: str,
    ollama_url: str,
    output: str | None,
    write: bool,
) -> None:
    """Generate README from codebase analysis."""
    import httpx
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn

    # Collect key files
    code_summary = []
    for pattern in ["*.py", "*.js", "*.ts", "*.go", "*.rs"]:
        for fpath in list(target_path.rglob(pattern))[:10]:
            if any(p in str(fpath) for p in ["node_modules", ".git", "__pycache__", "venv"]):
                continue
            try:
                content = fpath.read_text(errors="replace")[:2000]
                code_summary.append(f"### {fpath.relative_to(target_path)}\n```\n{content}\n```")
            except Exception:
                pass

    # Read existing README for context
    existing_readme = ""
    for readme_name in ["README.md", "readme.md", "README.rst"]:
        readme_path = target_path / readme_name
        if readme_path.exists():
            existing_readme = readme_path.read_text()[:3000]
            break

    prompt = f"""Generate a comprehensive README.md for this project.

Project directory: {target_path.name}

{"Existing README:" + chr(10) + existing_readme + chr(10) if existing_readme else ""}

Key source files:
{chr(10).join(code_summary[:5])}

Generate a complete README.md with:
1. Project title and description
2. Features list
3. Installation instructions
4. Quick start / usage examples
5. API reference (if applicable)
6. Contributing guidelines

Output only the README.md content in Markdown."""

    console.print()
    console.print(f"[bold]llmstack docs readme[/]  model=[cyan]{model}[/]")
    console.print()

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]Generating README..."), console=console
    ) as progress:
        task = progress.add_task("Generating", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        result = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": DOCS_SYSTEM_PROMPT},
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

    out_path = Path(output) if output else (target_path / "README.md")
    if write:
        out_path.write_text(result)
        console.print(f"[green]README saved to {out_path}[/]")
    else:
        console.print()
        console.print(Markdown(result))
        console.print()
        console.print(f"[dim]Tip: Use --write to save to {out_path}[/]")


async def _generate_docstrings(
    target_path: Path,
    model: str,
    ollama_url: str,
    output: str | None,
    write: bool,
) -> None:
    """Generate docstrings for Python files."""
    import httpx
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    from rich.syntax import Syntax

    # Find Python files
    if target_path.is_file():
        py_files = [target_path]
    else:
        py_files = [
            p
            for p in target_path.rglob("*.py")
            if not any(x in str(p) for x in ["__pycache__", ".git", "venv", "node_modules"])
        ]

    if not py_files:
        console.print("[warning]No Python files found.[/]")
        return

    console.print()
    console.print(f"[bold]llmstack docs[/]  model=[cyan]{model}[/]  files=[dim]{len(py_files)}[/]")
    console.print()

    timeout = httpx.Timeout(300, connect=10, read=300, write=30)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating docstrings..."),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=len(py_files))

        for fpath in py_files[:5]:  # Limit to 5 files for safety
            content = fpath.read_text(errors="replace")
            if len(content) > 8000:
                progress.advance(task)
                continue

            prompt = f"""Add Google-style docstrings to all public functions and classes in this Python file that are missing docstrings.

File: {fpath}

```python
{content}
```

Output the COMPLETE file with docstrings added. Output ONLY the Python code, no preamble or explanation."""

            result = ""
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": DOCS_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": True,
                    },
                ) as resp:
                    if resp.status_code == 200:
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

            # Extract code from markdown blocks
            if "```python" in result:
                result = result.split("```python", 1)[1].split("```", 1)[0].strip()
            elif "```" in result:
                result = result.split("```", 1)[1].split("```", 1)[0].strip()

            if result.strip():
                if write:
                    fpath.write_text(result)
                    console.print(f"  [green]Updated {fpath.relative_to(Path.cwd())}[/]")
                else:
                    console.print()
                    console.print(f"[bold]{fpath.relative_to(Path.cwd())}[/]")
                    console.print(
                        Syntax(result[:3000], "python", theme="monokai", line_numbers=True)
                    )

            progress.advance(task)

    if write:
        console.print(f"\n[green]Docstrings generated for {min(len(py_files), 5)} files.[/]")
    else:
        console.print("\n[dim]Tip: Use --write to save changes to files[/]")
