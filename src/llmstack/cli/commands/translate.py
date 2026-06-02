"""llmstack translate — AI-powered code translation between programming languages."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


TRANSLATE_SYSTEM_PROMPT = """You are an expert polyglot programmer. Translate the given source code to the target language.

Rules:
- Preserve the original logic and behavior exactly
- Use idiomatic patterns for the target language
- Keep comments translated and relevant
- Use proper type annotations where the target language supports them
- Handle language-specific patterns (e.g., error handling, async, generics)
- Output ONLY the translated code, no explanations

If there are language-specific constructs that don't have a direct equivalent,
use the closest idiomatic alternative and add a brief comment explaining the difference."""

LANGUAGE_EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "go": ".go", "rust": ".rs", "java": ".java", "cpp": ".cpp",
    "c": ".c", "ruby": ".rb", "php": ".php", "swift": ".swift",
    "kotlin": ".kt", "scala": ".scala", "dart": ".dart",
    "csharp": ".cs", "lua": ".lua", "elixir": ".ex",
    "haskell": ".hs", "zig": ".zig", "nim": ".nim",
}

EXTENSION_TO_LANG = {v: k for k, v in LANGUAGE_EXTENSIONS.items()}


def translate(
    file: str,
    to_lang: str,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output: str | None = None,
    write: bool = False,
) -> None:
    """Translate source code to another programming language."""
    asyncio.run(_translate_async(
        file=file, to_lang=to_lang, model=model,
        ollama_url=ollama_url, output=output, write=write,
    ))


def _detect_language(file_path: Path) -> str:
    """Detect source language from file extension."""
    ext = file_path.suffix.lower()
    return EXTENSION_TO_LANG.get(ext, "unknown")


async def _translate_async(
    file: str,
    to_lang: str,
    model: str,
    ollama_url: str,
    output: str | None,
    write: bool,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

    ollama_url = ollama_url.rstrip("/")
    to_lang = to_lang.lower().strip()

    # Normalize language name
    lang_aliases = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "rb": "ruby", "rs": "rust", "cs": "csharp", "c++": "cpp",
        "c#": "csharp", "kt": "kotlin",
    }
    to_lang = lang_aliases.get(to_lang, to_lang)

    if to_lang not in LANGUAGE_EXTENSIONS:
        console.print(f"[error]Unsupported target language: {to_lang}[/]")
        console.print(f"Supported: {', '.join(sorted(LANGUAGE_EXTENSIONS.keys()))}")
        raise typer.Exit(1)

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print("[error]Ollama is not responding.[/]")
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(Panel(
            "[error]Cannot connect to Ollama.[/]\n\nMake sure Ollama is running:\n  [bold cyan]ollama serve[/]",
            title="Connection Error", border_style="red",
        ))
        raise typer.Exit(1)

    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[error]File not found: {file}[/]")
        raise typer.Exit(1)

    source_code = file_path.read_text(errors="replace")
    from_lang = _detect_language(file_path)

    console.print()
    console.print(f"[bold]llmstack translate[/]  [cyan]{from_lang}[/] → [green]{to_lang}[/]  model=[dim]{model}[/]")
    console.print(f"  [dim]Source: {file_path} ({len(source_code)} chars)[/]")
    console.print()

    # Truncate very large files
    max_chars = 15000
    if len(source_code) > max_chars:
        source_code = source_code[:max_chars]
        console.print(f"  [warning]File truncated to {max_chars} chars[/]")

    prompt = f"""Translate this {from_lang} code to {to_lang}:

```{from_lang}
{source_code}
```

Output only the translated {to_lang} code, no explanations."""

    # Stream LLM response
    translated = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Translating..."),
        console=console,
    ) as progress:
        task = progress.add_task("Translating", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
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
                            translated += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    # Strip code fences if present
    if "```" in translated:
        lines = translated.split("\n")
        cleaned = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside or not any(line.strip().startswith("```") for _ in [1]):
                cleaned.append(line)
        translated = "\n".join(cleaned).strip()
        if not translated:
            # fallback: just strip first and last ``` lines
            translated = "\n".join(
                ln for ln in lines if not ln.strip().startswith("```")
            ).strip()

    # Display result
    syntax_map = {
        "python": "python", "javascript": "javascript", "typescript": "typescript",
        "go": "go", "rust": "rust", "java": "java", "cpp": "cpp", "c": "c",
        "ruby": "ruby", "php": "php", "swift": "swift", "kotlin": "kotlin",
        "csharp": "csharp", "lua": "lua", "dart": "dart",
    }
    console.print()
    console.print(Syntax(
        translated,
        syntax_map.get(to_lang, to_lang),
        theme="monokai",
        line_numbers=True,
    ))

    # Write output
    if write or output:
        out_path = Path(output) if output else file_path.with_suffix(LANGUAGE_EXTENSIONS[to_lang])
        out_path.write_text(translated + "\n")
        console.print()
        console.print(f"[green]Translated code written to {out_path}[/]")
