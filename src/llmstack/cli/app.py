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
    question: str = typer.Argument("", help="Question to ask (omit for interactive mode)"),
    files: list[Path] = typer.Argument(None, help="Files or directories to search"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="LLM model for generation"),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Embedding model"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of relevant chunks"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive conversation mode"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable persistent index, re-index from scratch"),
    git: bool = typer.Option(True, "--git/--no-git", help="Include git context (branch, recent commits)"),
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
        interactive=interactive or not question,
        no_cache=no_cache,
        use_git=git,
    )


@app.command(name="eval")
def eval_cmd(
    data: str = typer.Option(None, "--data", "-d", help="Path to evaluation dataset"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model to evaluate"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    gateway_url: str = typer.Option(None, "--gateway-url", "-g", help="Running gateway URL (shows live quality)"),
    max_examples: int = typer.Option(20, "--max-examples", help="Max eval examples"),
    output: str = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
) -> None:
    """Evaluate model quality against a test dataset or live gateway."""
    from llmstack.cli.commands.eval import eval_cmd as _eval
    _eval(
        data=data, model=model, ollama_url=ollama_url,
        gateway_url=gateway_url, max_examples=max_examples, output=output,
    )


@app.command()
def finetune(
    data: str = typer.Argument(..., help="Path to training data (CSV, JSON, JSONL, TXT, Parquet)"),
    base_model: str = typer.Option(
        "unsloth/llama-3.2-1b-instruct-bnb-4bit", "--base", "-b", help="Base model name or path",
    ),
    method: str = typer.Option("qlora", "--method", help="Training method: qlora, lora, full"),
    output: str = typer.Option("./finetune-output", "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(None, "--epochs", "-e", help="Number of training epochs (auto if unset)"),
    lr: float = typer.Option(None, "--lr", help="Learning rate (auto if unset)"),
    batch_size: int = typer.Option(None, "--batch-size", help="Batch size (auto if unset)"),
    lora_r: int = typer.Option(None, "--lora-r", help="LoRA rank (auto if unset)"),
    max_seq_length: int = typer.Option(2048, "--max-seq-length", help="Maximum sequence length"),
    eval_split: float = typer.Option(0.1, "--eval-split", help="Fraction of data for evaluation"),
    export_gguf: bool = typer.Option(False, "--export-gguf", help="Export model to GGUF format"),
    export_ollama: str = typer.Option(None, "--export-ollama", help="Create Ollama model with this name"),
    quantization: str = typer.Option("q4_k_m", "--quant", "-q", help="GGUF quantization: q4_k_m, q5_k_m, q8_0, f16"),
    system_prompt: str = typer.Option("", "--system", "-s", help="System prompt for all examples"),
    resume: str = typer.Option(None, "--resume", help="Resume from checkpoint path"),
) -> None:
    """Fine-tune a model on your data with LoRA/QLoRA. One command, zero boilerplate."""
    from llmstack.cli.commands.finetune import finetune as _finetune
    _finetune(
        data=data, base_model=base_model, method=method, output=output,
        epochs=epochs, lr=lr, batch_size=batch_size, lora_r=lora_r,
        max_seq_length=max_seq_length, eval_split=eval_split,
        export_gguf=export_gguf, export_ollama=export_ollama,
        quantization=quantization, system_prompt=system_prompt, resume=resume,
    )


@app.command(name="agent")
def agent_cmd(
    task: str = typer.Argument(..., help="Task for the agent to complete"),
    model: str = typer.Option(None, "--model", "-m", help="Model name (default: llama3.2)"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    max_steps: int = typer.Option(25, "--max-steps", help="Maximum agent steps"),
    tools: str = typer.Option(None, "--tools", "-t", help="Comma-separated tool names to enable"),
    working_dir: str = typer.Option(".", "--dir", "-d", help="Working directory for file operations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run an AI agent that uses tools to complete a task."""
    from llmstack.cli.commands.agent import agent as _agent
    _agent(
        task=task, model=model, ollama_url=ollama_url,
        max_steps=max_steps, tools=tools,
        working_dir=working_dir, verbose=verbose,
    )


@app.command(name="mcp")
def mcp_cmd(
    model: str = typer.Option(None, "--model", "-m", help="Model name (default: llama3.2)"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    working_dir: str = typer.Option(".", "--dir", "-d", help="Working directory"),
) -> None:
    """Start the MCP server for AI client integration (Claude Code, Cursor, etc.)."""
    from llmstack.cli.commands.mcp import mcp_serve as _mcp
    _mcp(model=model, ollama_url=ollama_url, working_dir=working_dir)
