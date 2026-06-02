"""llmstack diagram — Generate architecture diagrams from code using Mermaid."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from llmstack.cli.console import console


DIAGRAM_SYSTEM_PROMPT = """You are a software architect who creates clear, accurate diagrams from code.

Generate a Mermaid diagram based on the provided code structure.

Rules:
- Use appropriate Mermaid diagram type (flowchart, classDiagram, sequenceDiagram, erDiagram, etc.)
- Show key relationships between components
- Use clear, descriptive labels
- Keep diagrams readable (not too many nodes)
- Use appropriate styling and directions

Output ONLY the Mermaid diagram code, no explanations or markdown fences."""


DIAGRAM_TYPES = {
    "architecture": "Generate a high-level architecture diagram showing major components and their relationships",
    "class": "Generate a class diagram showing classes, their attributes, methods, and relationships",
    "sequence": "Generate a sequence diagram showing the main interaction flow",
    "flow": "Generate a flowchart showing the main logic flow",
    "er": "Generate an entity-relationship diagram for data models",
    "dependency": "Generate a dependency graph showing module imports and relationships",
    "state": "Generate a state diagram showing state transitions",
}


def diagram(
    target: str | None = None,
    diagram_type: str = "architecture",
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output: str | None = None,
) -> None:
    """Generate architecture diagrams from code."""
    asyncio.run(_diagram_async(
        target=target, diagram_type=diagram_type, model=model,
        ollama_url=ollama_url, output=output,
    ))


def _collect_structure(directory: Path) -> str:
    """Collect project structure for diagram generation."""
    code_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp"}
    ignore = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".tox", "dist", "build"}

    lines = []
    file_count = 0
    max_files = 30

    for p in sorted(directory.rglob("*")):
        if any(part in ignore for part in p.parts):
            continue
        if p.is_file() and p.suffix in code_exts and file_count < max_files:
            rel = p.relative_to(directory)
            lines.append(f"  {rel}")
            file_count += 1

            # Read first few lines to detect classes/functions
            try:
                content = p.read_text(errors="replace")
                for line in content.split("\n")[:100]:
                    stripped = line.strip()
                    if any(stripped.startswith(kw) for kw in
                           ["class ", "def ", "async def ", "function ", "func ", "fn ", "pub fn ",
                            "export ", "interface ", "type ", "struct "]):
                        lines.append(f"    → {stripped[:80]}")
            except OSError:
                pass

    return "\n".join(lines)


def _collect_imports(directory: Path) -> str:
    """Collect import relationships."""
    imports = []
    code_exts = {".py", ".js", ".ts"}
    ignore = {"__pycache__", ".git", "node_modules", "venv"}

    for p in sorted(directory.rglob("*")):
        if any(part in ignore for part in p.parts):
            continue
        if p.is_file() and p.suffix in code_exts:
            try:
                content = p.read_text(errors="replace")
                rel = str(p.relative_to(directory))
                for line in content.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ", "require(", "import {")):
                        imports.append(f"{rel}: {stripped[:100]}")
            except OSError:
                pass

    return "\n".join(imports[:200])


async def _diagram_async(
    target: str | None,
    diagram_type: str,
    model: str,
    ollama_url: str,
    output: str | None,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

    ollama_url = ollama_url.rstrip("/")

    if diagram_type not in DIAGRAM_TYPES:
        console.print(f"[error]Unknown diagram type: {diagram_type}[/]")
        console.print(f"Available: {', '.join(DIAGRAM_TYPES.keys())}")
        raise typer.Exit(1)

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

    target_path = Path(target) if target else Path.cwd()

    if target_path.is_file():
        content = target_path.read_text(errors="replace")[:12000]
        context = f"File: {target_path.name}\n\n```\n{content}\n```"
    else:
        structure = _collect_structure(target_path)
        imports = _collect_imports(target_path) if diagram_type == "dependency" else ""
        context = f"Project structure:\n{structure}"
        if imports:
            context += f"\n\nImport relationships:\n{imports}"

    console.print()
    console.print(f"[bold]llmstack diagram[/]  type=[cyan]{diagram_type}[/]  model=[dim]{model}[/]")
    console.print()

    type_instruction = DIAGRAM_TYPES[diagram_type]
    prompt = f"""{type_instruction}

{context}

Generate a Mermaid diagram. Output ONLY the Mermaid code, no explanations."""

    # Stream response
    mermaid_code = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating diagram..."),
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
                        {"role": "system", "content": DIAGRAM_SYSTEM_PROMPT},
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
                            mermaid_code += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    # Clean up code fences
    if "```mermaid" in mermaid_code:
        mermaid_code = mermaid_code.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    elif "```" in mermaid_code:
        mermaid_code = mermaid_code.split("```", 1)[1].split("```", 1)[0].strip()

    console.print()
    console.print(Syntax(mermaid_code, "text", theme="monokai", line_numbers=True))

    if output:
        out_path = Path(output)
        if out_path.suffix == ".md":
            out_path.write_text(f"```mermaid\n{mermaid_code}\n```\n")
        else:
            out_path.write_text(mermaid_code + "\n")
        console.print(f"\n[green]Diagram saved to {output}[/]")
    else:
        console.print()
        console.print("[dim]Tip: Copy the diagram to mermaid.live to render it visually[/]")
        console.print("[dim]Or save with: llmstack diagram --output diagram.md[/]")
