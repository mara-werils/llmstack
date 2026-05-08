"""llmstack ask — ask questions about local files using a local LLM."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from llmstack.cli.console import console


def ask(
    question: str = typer.Argument(..., help="Question to ask about your files"),
    files: Optional[list[Path]] = typer.Argument(None, help="Files or directories to search"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="LLM model for generation"),
    embed_model: str = typer.Option(
        "nomic-embed-text", "--embed-model", help="Embedding model"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of relevant chunks"),
    ollama_url: str = typer.Option(
        "http://localhost:11434", "--ollama-url", help="Ollama API URL"
    ),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Show source citations"
    ),
) -> None:
    """Ask questions about local files using a local LLM.

    Parses your files, creates embeddings, finds relevant context,
    and generates an answer — all locally with Ollama.
    """
    asyncio.run(_ask_async(
        question=question,
        files=files,
        model=model,
        embed_model=embed_model,
        top_k=top_k,
        ollama_url=ollama_url,
        show_sources=show_sources,
    ))


async def _ask_async(
    question: str,
    files: list[Path] | None,
    model: str,
    embed_model: str,
    top_k: int,
    ollama_url: str,
    show_sources: bool,
) -> None:
    """Async implementation of the ask command."""
    from llmstack.ask.engine import AskEngine
    from llmstack.ask.parsers import collect_files

    # ── Step 0: Handle stdin piping ──────────────────────────────────────
    stdin_content: str | None = None
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read()

    # ── Step 1: Check Ollama connectivity ────────────────────────────────
    ollama_url = ollama_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print(
                    "[error]Ollama is not responding correctly.[/]\n"
                    f"  Status: {resp.status_code}\n"
                    "  Make sure Ollama is running: [bold]ollama serve[/]"
                )
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(Panel(
            "[error]Cannot connect to Ollama.[/]\n\n"
            "Make sure Ollama is running:\n"
            "  [bold cyan]ollama serve[/]\n\n"
            "Or install it from: [link]https://ollama.ai[/link]\n\n"
            f"Tried connecting to: {ollama_url}",
            title="Connection Error",
            border_style="red",
        ))
        raise typer.Exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[error]Error connecting to Ollama: {exc}[/]")
        raise typer.Exit(1)

    console.print()
    console.print(
        f"[bold]llmstack ask[/]  model=[cyan]{model}[/]  "
        f"embeddings=[cyan]{embed_model}[/]"
    )
    console.print()

    # ── Step 2: Collect and parse files ──────────────────────────────────
    engine = AskEngine(
        ollama_url=ollama_url,
        model=model,
        embed_model=embed_model,
    )

    try:
        # Determine input paths
        paths: list[Path] = []
        if stdin_content:
            # Write stdin content to a temp-like in-memory approach:
            # we create TextChunks directly
            from llmstack.ask.parsers import TextChunk

            lines = stdin_content.splitlines()
            stdin_chunks = [TextChunk(
                content=stdin_content.strip(),
                source="<stdin>",
                start_line=1,
                end_line=max(len(lines), 1),
            )]
            # Manually load stdin chunks + any file paths
            if files:
                for f in files:
                    paths.extend(collect_files(f))
        elif files:
            for f in files:
                paths.extend(collect_files(f))
        else:
            # Default to current directory
            paths = collect_files(Path.cwd())

        if not paths and not stdin_content:
            console.print("[warning]No supported files found.[/]")
            raise typer.Exit(1)

        # Parse files with progress
        total_files = len(paths)
        if total_files > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Parsing files..."),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Parsing", total=total_files)

                all_chunks: list[TextChunk] = []
                for fpath in paths:
                    try:
                        from llmstack.ask.parsers import parse_file

                        chunks = parse_file(fpath)
                        all_chunks.extend(chunks)
                    except Exception:
                        pass
                    progress.advance(task)

            # Add stdin chunks if any
            if stdin_content:
                all_chunks = stdin_chunks + all_chunks

            engine._chunks = all_chunks
        else:
            # stdin only
            all_chunks = stdin_chunks
            engine._chunks = all_chunks

        if not engine._chunks:
            console.print("[warning]No content could be extracted from the files.[/]")
            raise typer.Exit(1)

        chunk_count = len(engine._chunks)
        console.print(
            f"  [dim]Found [bold]{total_files}[/bold] files, "
            f"[bold]{chunk_count}[/bold] chunks[/]"
        )

        # ── Step 3: Index chunks ─────────────────────────────────────────
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Indexing {chunk_count} chunks..."),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing", total=None)
            await engine.embeddings.index(engine._chunks)
            progress.update(task, completed=True)

        # ── Step 4: Search and generate answer ───────────────────────────
        console.print()
        console.print(
            f"  [dim]Searching for relevant context (top {top_k})...[/]"
        )

        # Get search results for source display
        results = await engine.embeddings.search(question, top_k=top_k)

        if not results:
            console.print("[warning]No relevant context found.[/]")
            raise typer.Exit(1)

        console.print()
        console.print("[bold green]Answer:[/]")
        console.print()

        # Stream the answer
        answer_text = ""
        async for token in engine.ask(question, top_k=top_k):
            print(token, end="", flush=True)
            answer_text += token
        print()  # newline after streamed response

        # ── Step 5: Show source citations ────────────────────────────────
        if show_sources and results:
            console.print()
            source_table = Table(
                title="Sources",
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                padding=(0, 1),
            )
            source_table.add_column("File", style="bold")
            source_table.add_column("Lines", style="dim")
            source_table.add_column("Relevance", justify="right")
            source_table.add_column("Preview", style="dim", max_width=60)

            for chunk, score in results:
                # Shorten file path for display
                try:
                    display_path = str(Path(chunk.source).relative_to(Path.cwd()))
                except ValueError:
                    display_path = chunk.source

                preview = chunk.content[:80].replace("\n", " ")
                relevance_color = (
                    "green" if score >= 0.8
                    else "yellow" if score >= 0.6
                    else "red"
                )
                source_table.add_row(
                    display_path,
                    f"{chunk.start_line}-{chunk.end_line}",
                    f"[{relevance_color}]{score:.2f}[/]",
                    preview,
                )

            console.print(source_table)

        console.print()

    except typer.Exit:
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(Panel(
                f"[error]Model not found.[/]\n\n"
                f"The model [bold]{model}[/bold] or [bold]{embed_model}[/bold] "
                f"may not be available.\n\n"
                f"Pull it with:\n"
                f"  [bold cyan]ollama pull {model}[/]\n"
                f"  [bold cyan]ollama pull {embed_model}[/]",
                title="Model Error",
                border_style="red",
            ))
        else:
            console.print(f"[error]HTTP error: {exc}[/]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[error]Unexpected error: {exc}[/]")
        raise typer.Exit(1)
    finally:
        await engine.close()
