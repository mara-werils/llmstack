"""CLI entry point — Typer application."""

from __future__ import annotations

from pathlib import Path

import typer

from llmstack import __version__


app = typer.Typer(
    name="llmstack",
    help="One command. Full LLM stack. Zero config.",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llmstack {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """LLMStack — One command. Full LLM stack. Zero config."""


@app.command()
def init(
    preset: str = typer.Option(None, "--preset", "-p", help="Preset: chat, rag, agent"),
    directory: str = typer.Option(None, "--dir", "-d", help="Target directory"),
) -> None:
    """Create a new llmstack.yaml configuration file."""
    from pathlib import Path
    from llmstack.cli.commands.init import init as _init
    _init(preset=preset, directory=Path(directory) if directory else None)


@app.command()
def up(
    attach: bool = typer.Option(False, "--attach", "-a", help="Stream logs after starting"),
) -> None:
    """Start all services defined in llmstack.yaml."""
    from llmstack.cli.commands.up import up as _up
    _up(attach=attach)


@app.command()
def down(
    volumes: bool = typer.Option(False, "--volumes", "-v", help="Also remove data volumes"),
) -> None:
    """Stop and remove all llmstack services."""
    from llmstack.cli.commands.down import down as _down
    _down(volumes=volumes)


@app.command()
def status() -> None:
    """Show the status of all running llmstack services."""
    from llmstack.cli.commands.status import status as _status
    _status()


@app.command()
def logs(
    service: str = typer.Argument(help="Service name (ollama, qdrant, redis)"),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f"),
    tail: int = typer.Option(50, "--tail", "-n"),
) -> None:
    """Stream logs from a specific service."""
    from llmstack.cli.commands.logs import logs as _logs
    _logs(service=service, follow=follow, tail=tail)


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Model name to chat with"),
) -> None:
    """Interactive chat with your local LLM."""
    from llmstack.cli.commands.chat import chat as _chat
    _chat(model=model)


@app.command()
def export(
    output: str = typer.Option("docker-compose.yml", "--output", "-o", help="Output file path"),
) -> None:
    """Export llmstack.yaml as a standalone docker-compose.yml."""
    from llmstack.cli.commands.export import export as _export
    _export(output=output)


@app.command()
def doctor() -> None:
    """Check system requirements and diagnose issues."""
    from llmstack.cli.commands.doctor import doctor as _doctor
    _doctor()


@app.command()
def bench(
    model: str = typer.Option(None, "--model", "-m", help="Model name(s), comma-separated"),
    suite: str = typer.Option("all", "--suite", "-s", help="Benchmark suite(s): simple,reasoning,coding,long_context,creative,all"),
    output: str = typer.Option(None, "--output", "-o", help="Export results to JSON file"),
) -> None:
    """Benchmark models and show comparative performance results."""
    from llmstack.cli.commands.bench import bench as _bench
    _bench(model=model, suite=suite, output=output)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about your files"),
    files: list[Path] = typer.Argument(None, help="Files or directories to search"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="LLM model for generation"),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Embedding model"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of relevant chunks"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
) -> None:
    """Ask questions about local files using a local LLM."""
    from llmstack.cli.commands.ask import ask as _ask
    _ask(
        question=question,
        files=files,
        model=model,
        embed_model=embed_model,
        top_k=top_k,
        ollama_url=ollama_url,
        show_sources=show_sources,
    )
