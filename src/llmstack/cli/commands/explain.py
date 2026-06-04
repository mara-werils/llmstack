"""llmstack explain — Deep code explanation with examples and diagrams."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


EXPLAIN_SYSTEM_PROMPT = """You are an expert software educator. Explain the provided code clearly and thoroughly.

Your explanation should include:
1. **Overview**: What the code does in one sentence
2. **Architecture**: How the components fit together
3. **Key Concepts**: Important patterns, algorithms, or techniques used
4. **Step-by-Step Walkthrough**: Walk through the logic flow
5. **Complexity**: Time and space complexity if applicable
6. **Potential Issues**: Edge cases, bugs, or improvements
7. **Mermaid Diagram**: If the code has complex flow or architecture, include a Mermaid diagram

Format your response in Markdown. Be thorough but accessible — explain as if to a mid-level developer
who is new to this codebase."""


EXPLAIN_FUNCTION_PROMPT = """You are an expert software educator. Explain the specific function/class provided.

Include:
1. **Purpose**: What it does and why
2. **Parameters**: Each parameter explained
3. **Return Value**: What it returns
4. **Logic Flow**: Step-by-step walkthrough
5. **Usage Example**: Show how to call/use it
6. **Edge Cases**: Important edge cases to be aware of

Format in Markdown."""


def explain(
    target: str,
    symbol: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    level: str = "mid",
    output: str | None = None,
) -> None:
    """Explain code in detail with diagrams and examples."""
    asyncio.run(
        _explain_async(
            target=target,
            symbol=symbol,
            model=model,
            ollama_url=ollama_url,
            level=level,
            output=output,
        )
    )


def _extract_symbol(content: str, symbol: str, file_ext: str) -> str | None:
    """Try to extract a specific function/class from source code."""
    lines = content.split("\n")
    result_lines = []
    found = False
    indent_level = -1

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Python-style
        if file_ext == ".py":
            if stripped.startswith(("def ", "class ", "async def ")) and symbol in stripped:
                found = True
                indent_level = len(line) - len(line.lstrip())
                result_lines.append(line)
                continue
            if found:
                if stripped == "" or stripped.startswith("#"):
                    result_lines.append(line)
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent > indent_level or stripped == "":
                    result_lines.append(line)
                else:
                    break

        # JS/TS/Go/Rust/Java-style (brace-based)
        elif file_ext in (".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"):
            if symbol in stripped and any(
                kw in stripped
                for kw in [
                    "function ",
                    "func ",
                    "fn ",
                    "def ",
                    "class ",
                    "pub ",
                    "private ",
                    "public ",
                    "const ",
                ]
            ):
                found = True
                brace_count = 0
            if found:
                result_lines.append(line)
                brace_count += stripped.count("{") - stripped.count("}")
                if brace_count <= 0 and len(result_lines) > 1:
                    break

    return "\n".join(result_lines) if result_lines else None


async def _explain_async(
    target: str,
    symbol: str | None,
    model: str,
    ollama_url: str,
    level: str,
    output: str | None,
) -> None:
    import httpx
    import typer
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
        console.print(
            Panel(
                "[error]Cannot connect to Ollama.[/]",
                title="Connection Error",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    target_path = Path(target)
    if not target_path.exists():
        console.print(f"[error]File not found: {target}[/]")
        raise typer.Exit(1)

    content = target_path.read_text(errors="replace")

    # Extract specific symbol if requested
    code_to_explain = content
    if symbol:
        extracted = _extract_symbol(content, symbol, target_path.suffix)
        if extracted:
            code_to_explain = extracted
            console.print(f"  [dim]Explaining symbol: {symbol}[/]")
        else:
            console.print(f"[warning]Symbol '{symbol}' not found, explaining entire file.[/]")

    # Truncate if too large
    max_chars = 12000
    if len(code_to_explain) > max_chars:
        code_to_explain = code_to_explain[:max_chars] + "\n... (truncated)"

    level_desc = {
        "beginner": "Explain as if to a junior developer or student. Define all technical terms.",
        "mid": "Explain as if to a mid-level developer who is new to this codebase.",
        "senior": "Explain concisely, focusing on architecture decisions and non-obvious patterns.",
    }

    console.print()
    console.print(f"[bold]llmstack explain[/]  model=[cyan]{model}[/]  level=[dim]{level}[/]")
    console.print(f"  [dim]File: {target_path} ({len(code_to_explain)} chars)[/]")
    console.print()

    system_prompt = EXPLAIN_FUNCTION_PROMPT if symbol else EXPLAIN_SYSTEM_PROMPT
    prompt = f"""{level_desc.get(level, level_desc["mid"])}

File: {target_path.name}
Language: {target_path.suffix}

```
{code_to_explain}
```

Provide a thorough explanation."""

    # Stream response
    explanation = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Analyzing code..."),
        console=console,
    ) as progress:
        task = progress.add_task("Explaining", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
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
                            explanation += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    console.print()
    console.print(Markdown(explanation))

    if output:
        Path(output).write_text(explanation + "\n")
        console.print(f"\n[green]Explanation saved to {output}[/]")
