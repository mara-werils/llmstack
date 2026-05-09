"""llmstack ask — ask questions about local files using a local LLM.

v2: persistent index, AST chunking, hybrid search, git context, interactive mode.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from llmstack.cli.console import console


def ask(
    question: str = "",
    files: list[Path] | None = None,
    model: str = "llama3.2",
    embed_model: str = "nomic-embed-text",
    top_k: int = 5,
    ollama_url: str = "http://localhost:11434",
    show_sources: bool = True,
    interactive: bool = False,
    no_cache: bool = False,
    use_git: bool = True,
) -> None:
    """Ask questions about local files using a local LLM."""
    asyncio.run(_ask_async(
        question=question, files=files, model=model, embed_model=embed_model,
        top_k=top_k, ollama_url=ollama_url, show_sources=show_sources,
        interactive=interactive, no_cache=no_cache, use_git=use_git,
    ))


async def _ask_async(
    question: str,
    files: list[Path] | None,
    model: str,
    embed_model: str,
    top_k: int,
    ollama_url: str,
    show_sources: bool,
    interactive: bool,
    no_cache: bool,
    use_git: bool,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn,
    )

    from llmstack.ask.parsers import TextChunk, collect_files, parse_file
    from llmstack.ask.ast_chunker import chunk_code
    from llmstack.ask.embeddings import LocalEmbeddings
    from llmstack.ask.hybrid_search import HybridSearcher
    from llmstack.ask.index import PersistentIndex

    # ── Check Ollama ─────────────────────────────────────────────────────
    ollama_url = ollama_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print("[error]Ollama is not responding correctly.[/]")
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(Panel(
            "[error]Cannot connect to Ollama.[/]\n\n"
            "Make sure Ollama is running:\n  [bold cyan]ollama serve[/]\n\n"
            f"Tried: {ollama_url}",
            title="Connection Error", border_style="red",
        ))
        raise typer.Exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[error]Error connecting to Ollama: {exc}[/]")
        raise typer.Exit(1)

    console.print()
    console.print(
        f"[bold]llmstack ask[/]  model=[cyan]{model}[/]  embeddings=[cyan]{embed_model}[/]"
    )

    # ── Handle stdin ─────────────────────────────────────────────────────
    stdin_chunks: list[TextChunk] = []
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read()
        if stdin_content.strip():
            lines = stdin_content.splitlines()
            stdin_chunks = [TextChunk(
                content=stdin_content.strip(),
                source="<stdin>",
                start_line=1,
                end_line=max(len(lines), 1),
            )]

    # ── Collect files ────────────────────────────────────────────────────
    paths: list[Path] = []
    if files:
        for f in files:
            paths.extend(collect_files(f))
    elif not stdin_chunks:
        paths = collect_files(Path.cwd())

    if not paths and not stdin_chunks:
        console.print("[warning]No supported files found.[/]")
        raise typer.Exit(1)

    # ── Determine project root for persistent index ──────────────────────
    project_root = Path.cwd()
    if files and len(files) == 1 and files[0].is_dir():
        project_root = files[0].resolve()

    # ── Git context ──────────────────────────────────────────────────────
    git_context_text = ""
    if use_git:
        from llmstack.ask.git_context import get_git_info
        git_info = get_git_info(project_root)
        if git_info.is_repo:
            git_context_text = git_info.to_context()
            console.print(f"  [dim]Git: {git_info.branch} ({len(git_info.recent_commits)} recent commits)[/]")

    # ── Persistent index: check what changed ─────────────────────────────
    index = PersistentIndex(project_root) if not no_cache else None
    use_cached = False
    files_to_parse = paths

    if index and index.exists() and paths:
        to_update, _, removed = index.diff(paths)

        if removed:
            index.remove_files(removed)

        if not to_update:
            use_cached = True
            console.print(f"  [dim]Index cached: {index.chunk_count()} chunks (0 files changed)[/]")
        else:
            console.print(
                f"  [dim]Index: {len(to_update)} files changed, "
                f"{len(paths) - len(to_update)} cached[/]"
            )
            files_to_parse = to_update

    # ── Parse files with AST-aware chunking ──────────────────────────────
    all_chunks: list[TextChunk] = []

    if use_cached and index:
        all_chunks = index.load_chunks()
    else:
        # Load cached chunks for unchanged files
        cached_chunks: list[TextChunk] = []
        if index and index.exists():
            cached_chunks = index.load_chunks()
            # Keep chunks for files NOT in files_to_parse
            parse_set = {str(p.resolve()) for p in files_to_parse}
            try:
                cached_chunks = [
                    c for c in cached_chunks
                    if str((project_root / c.source).resolve()) not in parse_set
                ]
            except Exception:
                cached_chunks = []

        # Parse changed files
        new_chunks: list[TextChunk] = []
        new_file_chunks: dict[str, list[TextChunk]] = {}
        file_hashes: dict[str, str] = {}

        if files_to_parse:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Parsing files..."),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Parsing", total=len(files_to_parse))

                for fpath in files_to_parse:
                    try:
                        # Use AST chunker for code files
                        ext = fpath.suffix.lower()
                        code_exts = {
                            ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
                            ".java", ".c", ".cpp", ".h", ".rb",
                        }
                        if ext in code_exts:
                            source = fpath.read_text(errors="replace")
                            chunks = chunk_code(source, str(fpath))
                        else:
                            chunks = parse_file(fpath)

                        new_chunks.extend(chunks)

                        # Track for index update
                        rel_path = str(fpath.resolve().relative_to(project_root))
                        new_file_chunks[rel_path] = chunks
                        from llmstack.ask.index import _file_hash
                        file_hashes[rel_path] = _file_hash(fpath)
                    except Exception:
                        pass
                    progress.advance(task)

        all_chunks = cached_chunks + new_chunks + stdin_chunks

        if not all_chunks:
            console.print("[warning]No content could be extracted.[/]")
            raise typer.Exit(1)

        console.print(f"  [dim]Found {len(paths)} files, {len(all_chunks)} chunks[/]")

    # ── Index embeddings ─────────────────────────────────────────────────
    embeddings = LocalEmbeddings(ollama_url=ollama_url, model=embed_model)
    searcher = HybridSearcher()
    emb_array = None

    if use_cached and index:
        emb_array = index.load_embeddings()

    if emb_array is not None and len(emb_array) == len(all_chunks):
        console.print("  [dim]Embeddings loaded from cache[/]")
        searcher.index(all_chunks, emb_array)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Indexing {len(all_chunks)} chunks..."),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing", total=None)
            await embeddings.index(all_chunks)
            emb_array = embeddings._embeddings
            progress.update(task, completed=True)

        searcher.index(all_chunks, emb_array)

        # Save to persistent index
        if index and new_file_chunks:
            index.update(new_file_chunks, emb_array, file_hashes, all_chunks)
            console.print(f"  [dim]Index saved to {project_root / '.llmstack-index'}[/]")

    # ── Interactive or single-shot mode ──────────────────────────────────
    if interactive:
        await _interactive_loop(
            searcher=searcher, embeddings=embeddings, all_chunks=all_chunks,
            model=model, ollama_url=ollama_url, top_k=top_k,
            show_sources=show_sources, git_context=git_context_text,
        )
    else:
        await _single_query(
            question=question, searcher=searcher, embeddings=embeddings,
            model=model, ollama_url=ollama_url, top_k=top_k,
            show_sources=show_sources, git_context=git_context_text,
        )

    await embeddings.close()
    if index:
        index.close()


async def _single_query(
    question: str,
    searcher,
    embeddings,
    model: str,
    ollama_url: str,
    top_k: int,
    show_sources: bool,
    git_context: str,
) -> None:
    """Handle a single question."""
    import httpx
    import json
    from llmstack.ask.engine import _build_context, _PROMPT_TEMPLATE

    # Search
    query_emb = await embeddings.embed([question])
    query_vec = query_emb[0] if len(query_emb) > 0 else None
    results = searcher.search(question, query_embedding=query_vec, top_k=top_k)

    if not results:
        console.print("[warning]No relevant context found.[/]")
        return

    console.print()
    console.print("[bold green]Answer:[/]")
    console.print()

    # Build prompt with context + optional git info
    context = _build_context(results)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    if git_context:
        prompt = f"Git context:\n{git_context}\n\n{prompt}"

    # Stream response
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", f"{ollama_url}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    print()

    # Sources
    if show_sources and results:
        _print_sources(results)

    console.print()


async def _interactive_loop(
    searcher,
    embeddings,
    all_chunks,
    model: str,
    ollama_url: str,
    top_k: int,
    show_sources: bool,
    git_context: str,
) -> None:
    """Interactive multi-turn conversation mode."""
    from rich.panel import Panel
    from llmstack.ask.conversation import ConversationEngine

    conv = ConversationEngine(
        ollama_url=ollama_url, model=model, git_context=git_context,
    )

    console.print()
    console.print(Panel(
        "[bold]Interactive mode[/] — chat with your codebase\n"
        "[dim]Commands: /clear (reset), /sources (show last sources), /quit (exit)[/]",
        border_style="cyan",
    ))

    try:
        while True:
            console.print()
            try:
                question = console.input("[bold cyan]You:[/] ")
            except (EOFError, KeyboardInterrupt):
                break

            question = question.strip()
            if not question:
                continue

            if question.lower() in ("/quit", "/exit", "exit", "quit"):
                break

            if question.lower() == "/clear":
                conv.clear()
                console.print("[dim]Conversation cleared.[/]")
                continue

            if question.lower() == "/sources":
                if conv.history:
                    last_asst = [t for t in conv.history if t.role == "assistant"]
                    if last_asst and last_asst[-1].sources:
                        for src in last_asst[-1].sources:
                            console.print(f"  [dim]{src}[/]")
                    else:
                        console.print("[dim]No sources from last answer.[/]")
                continue

            # Search for context
            query_emb = await embeddings.embed([question])
            query_vec = query_emb[0] if len(query_emb) > 0 else None
            results = searcher.search(question, query_embedding=query_vec, top_k=top_k)

            console.print()
            console.print("[bold green]Assistant:[/]")
            console.print()

            async for token in conv.ask(question, results):
                print(token, end="", flush=True)
            print()

            if show_sources and results:
                _print_sources(results)

    finally:
        await conv.close()

    console.print("\n[dim]Goodbye![/]")


def _print_sources(results) -> None:
    """Display source citations table."""
    from rich.table import Table

    console.print()
    table = Table(
        title="Sources", show_header=True,
        header_style="bold cyan", border_style="dim", padding=(0, 1),
    )
    table.add_column("File", style="bold")
    table.add_column("Lines", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Preview", style="dim", max_width=60)

    for chunk, score in results:
        try:
            display_path = str(Path(chunk.source).relative_to(Path.cwd()))
        except ValueError:
            display_path = chunk.source

        preview = chunk.content[:80].replace("\n", " ")
        color = "green" if score >= 0.01 else "yellow" if score >= 0.005 else "red"
        table.add_row(
            display_path,
            f"{chunk.start_line}-{chunk.end_line}",
            f"[{color}]{score:.4f}[/]",
            preview,
        )

    console.print(table)
