"""llmstack test — AI-powered test case generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


TEST_SYSTEM_PROMPT = """You are an expert software engineer specializing in testing. Generate comprehensive test cases.

For Python: use pytest with descriptive test names.
For JavaScript/TypeScript: use Jest with describe/it blocks.

Test structure:
- Test happy path
- Test edge cases (empty input, None, zero, etc.)
- Test error cases
- Use mocking where needed (unittest.mock for Python)
- Include fixtures where appropriate

Output ONLY the test code, no explanations."""


def test_gen(
    target: str | None = None,
    output: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    framework: str = "pytest",
    write: bool = False,
    coverage: bool = False,
) -> None:
    """Generate test cases for code."""
    asyncio.run(_test_gen_async(
        target=target, output=output, model=model, ollama_url=ollama_url,
        framework=framework, write=write, coverage=coverage,
    ))


async def _test_gen_async(
    target: str | None,
    output: str | None,
    model: str,
    ollama_url: str,
    framework: str,
    write: bool,
    coverage: bool,
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
        console.print(Panel("[error]Cannot connect to Ollama.[/]", border_style="red"))
        raise typer.Exit(1)

    target_path = Path(target) if target else Path.cwd()

    if target_path.is_file():
        files_to_test = [target_path]
    else:
        files_to_test = [
            p for p in target_path.rglob("*.py")
            if not any(x in str(p) for x in ["test_", "_test", "__pycache__", ".git", "venv"])
        ][:5]

    if not files_to_test:
        console.print("[warning]No source files found to generate tests for.[/]")
        return

    console.print()
    console.print(f"[bold]llmstack test[/]  model=[cyan]{model}[/]  framework=[cyan]{framework}[/]")
    console.print()

    timeout = httpx.Timeout(300, connect=10, read=300, write=30)

    for fpath in files_to_test:
        content = fpath.read_text(errors="replace")
        if len(content) > 8000:
            console.print(f"[dim]Skipping {fpath} (too large)[/]")
            continue

        prompt = f"""Generate comprehensive {framework} tests for this Python file.

File: {fpath.name}

```python
{content}
```

Generate a complete test file with:
1. Imports
2. Fixtures (if needed)
3. Tests for ALL public functions and classes
4. Edge cases and error conditions

Output ONLY the test code."""

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Generating tests for {fpath.name}..."),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=None)

            result = ""
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": TEST_SYSTEM_PROMPT},
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

        # Extract code
        if "```python" in result:
            result = result.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in result:
            result = result.split("```", 1)[1].split("```", 1)[0].strip()

        if not result.strip():
            console.print(f"[warning]No tests generated for {fpath.name}[/]")
            continue

        # Determine output path
        test_name = f"test_{fpath.stem}.py"
        if output:
            out_path = Path(output)
        else:
            # Try to find tests/ directory
            tests_dir = fpath.parent.parent / "tests" / "unit"
            if not tests_dir.exists():
                tests_dir = fpath.parent.parent / "tests"
            if not tests_dir.exists():
                tests_dir = fpath.parent
            out_path = tests_dir / test_name

        if write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result)
            console.print(f"[green]Tests saved to {out_path}[/]")
        else:
            console.print()
            console.print(f"[bold]{test_name}[/]")
            console.print(Syntax(result[:4000], "python", theme="monokai", line_numbers=True))
            console.print()
            console.print(f"[dim]Tip: Use --write to save to {out_path}[/]")
