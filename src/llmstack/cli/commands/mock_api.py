"""llmstack mock — Generate mock API servers from OpenAPI specs or descriptions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


MOCK_SYSTEM_PROMPT = """You are an API expert. Generate a complete FastAPI mock server from the API specification.

Output a Python file that:
1. Uses FastAPI with proper type annotations
2. Returns realistic mock data using Pydantic models
3. Includes all endpoints with proper HTTP methods
4. Has realistic response schemas
5. Includes basic error responses (404, 400, 500)
6. Has proper CORS middleware
7. Is immediately runnable with `uvicorn`

Output ONLY the Python code, no explanations."""


def mock_api(
    spec: str | None = None,
    description: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output: str = "mock_server.py",
    port: int = 9000,
) -> None:
    """Generate a mock API server."""
    asyncio.run(_mock_async(
        spec=spec, description=description, model=model,
        ollama_url=ollama_url, output=output, port=port,
    ))


async def _mock_async(
    spec: str | None,
    description: str | None,
    model: str,
    ollama_url: str,
    output: str,
    port: int,
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

    # Build prompt
    if spec:
        spec_path = Path(spec)
        if spec_path.exists():
            spec_content = spec_path.read_text(errors="replace")
            prompt = f"""Generate a FastAPI mock server from this API specification:

{spec_content[:10000]}

Create mock endpoints with realistic fake data for all paths."""
        else:
            console.print(f"[error]Spec file not found: {spec}[/]")
            raise typer.Exit(1)
    elif description:
        prompt = f"""Generate a FastAPI mock server for this API:

{description}

Create proper endpoints, Pydantic models, and realistic mock data.
Include pagination, filtering, CRUD operations where appropriate."""
    else:
        console.print("[error]Provide --spec (OpenAPI file) or --description[/]")
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold]llmstack mock[/]  model=[cyan]{model}[/]  port=[dim]{port}[/]")
    console.print()

    # Generate mock server
    server_code = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating mock API server..."),
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
                        {"role": "system", "content": MOCK_SYSTEM_PROMPT},
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
                            server_code += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    # Clean up code fences
    if "```python" in server_code:
        server_code = server_code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in server_code:
        server_code = server_code.split("```", 1)[1].split("```", 1)[0].strip()

    # Add runner if not present
    if "uvicorn" not in server_code:
        server_code += f'\n\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run(app, host="0.0.0.0", port={port})\n'

    # Display
    console.print()
    console.print(Syntax(server_code, "python", theme="monokai", line_numbers=True))

    # Write output
    out_path = Path(output)
    out_path.write_text(server_code)
    console.print()
    console.print(f"[green]Mock server written to {output}[/]")
    console.print(f"[dim]Run it: python {output}[/]")
    console.print(f"[dim]Or: uvicorn {out_path.stem}:app --port {port} --reload[/]")
