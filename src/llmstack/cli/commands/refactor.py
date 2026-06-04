"""llmstack refactor — AI-powered code refactoring suggestions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


REFACTOR_SYSTEM_PROMPT = """You are an expert software engineer specializing in code refactoring.

Analyze the code and suggest refactoring improvements. For each suggestion, output a JSON line:
{"priority": "HIGH|MEDIUM|LOW", "category": "pattern", "location": "function/class name", "current": "description of current code", "suggested": "what to change", "benefit": "why this improves the code", "effort": "small|medium|large"}

Categories:
- extract_method: Long methods that should be broken up
- rename: Unclear naming
- reduce_complexity: High cyclomatic complexity
- remove_duplication: DRY violations
- simplify_conditionals: Complex if/else chains
- improve_types: Missing or weak type annotations
- design_pattern: Applicable design patterns
- performance: Performance improvements
- error_handling: Better error handling patterns
- modernize: Use modern language features

After all suggestions, output:
{"type": "summary", "total": 5, "code_health_score": 7.5, "top_priority": "description of most impactful change"}

Be practical — suggest changes that meaningfully improve the code, not trivial style preferences."""


REFACTOR_STRATEGIES = {
    "clean": "Focus on clean code principles: naming, function length, SRP, DRY",
    "performance": "Focus on performance: algorithmic improvements, caching, lazy evaluation",
    "type-safety": "Focus on type safety: add types, remove Any, use generics, validate inputs",
    "solid": "Focus on SOLID principles: single responsibility, open/closed, dependency inversion",
    "testability": "Focus on testability: dependency injection, pure functions, smaller units",
}


def refactor(
    target: str,
    strategy: str = "clean",
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output: str | None = None,
    apply: bool = False,
) -> None:
    """Get AI-powered refactoring suggestions."""
    asyncio.run(
        _refactor_async(
            target=target,
            strategy=strategy,
            model=model,
            ollama_url=ollama_url,
            output=output,
            apply=apply,
        )
    )


async def _refactor_async(
    target: str,
    strategy: str,
    model: str,
    ollama_url: str,
    output: str | None,
    apply: bool,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.table import Table
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

    target_path = Path(target)
    if not target_path.exists():
        console.print(f"[error]File not found: {target}[/]")
        raise typer.Exit(1)

    content = target_path.read_text(errors="replace")
    strategy_desc = REFACTOR_STRATEGIES.get(strategy, REFACTOR_STRATEGIES["clean"])

    console.print()
    console.print(
        f"[bold]llmstack refactor[/]  model=[cyan]{model}[/]  strategy=[dim]{strategy}[/]"
    )
    console.print(f"  [dim]File: {target_path} ({len(content)} chars)[/]")
    console.print()

    max_chars = 12000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncated)"

    prompt = f"""Analyze this code and suggest refactoring improvements.

Strategy focus: {strategy_desc}

File: {target_path.name}

```
{content}
```

Output each suggestion as a JSON object on its own line, then a summary JSON object."""

    # Stream response
    full_response = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Analyzing for refactoring opportunities..."),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": REFACTOR_SYSTEM_PROMPT},
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

    # Parse suggestions
    suggestions = []
    summary_data = None

    for line in full_response.splitlines():
        line = line.strip()
        if line.startswith("```"):
            continue
        brace_idx = line.find("{")
        if brace_idx == -1:
            continue
        try:
            obj = json.loads(line[brace_idx:])
            if obj.get("type") == "summary":
                summary_data = obj
            elif "category" in obj or "priority" in obj:
                suggestions.append(obj)
        except json.JSONDecodeError:
            pass

    # Display results
    priority_colors = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "cyan"}
    priority_icons = {"HIGH": "!", "MEDIUM": "~", "LOW": "·"}

    if suggestions:
        table = Table(
            title="Refactoring Suggestions",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("Priority", width=8)
        table.add_column("Category", width=18)
        table.add_column("Location", width=20)
        table.add_column("Suggestion")
        table.add_column("Effort", width=8)

        for s in suggestions:
            p = s.get("priority", "LOW")
            color = priority_colors.get(p, "white")
            icon = priority_icons.get(p, "·")
            table.add_row(
                f"[{color}]{icon} {p}[/]",
                s.get("category", ""),
                s.get("location", ""),
                s.get("suggested", s.get("current", "")),
                s.get("effort", ""),
            )

        console.print()
        console.print(table)

    if summary_data:
        score = summary_data.get("code_health_score", 0)
        score_color = "green" if score >= 7 else "yellow" if score >= 4 else "red"
        console.print()
        console.print(
            Panel(
                f"[bold]Code Health Score:[/] [{score_color}]{score}/10[/]\n\n"
                f"[bold]Top Priority:[/] {summary_data.get('top_priority', 'N/A')}\n"
                f"[dim]Total suggestions: {summary_data.get('total', len(suggestions))}[/]",
                title="Refactoring Summary",
                border_style=score_color,
            )
        )

    if not suggestions and not summary_data:
        console.print()
        console.print(Panel(full_response, title="Refactoring Analysis", border_style="cyan"))

    # Save output
    if output:
        data = {
            "file": str(target_path),
            "strategy": strategy,
            "suggestions": suggestions,
            "summary": summary_data,
        }
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Report saved to {output}[/]")
