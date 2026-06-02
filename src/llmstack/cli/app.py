"""CLI entry point — Typer application."""

from __future__ import annotations

from pathlib import Path

import typer

from llmstack import __version__
from llmstack.cli.console import console


app = typer.Typer(
    name="llmstack",
    help="One command. Full LLM stack. Zero config.",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        import platform
        import sys
        typer.echo(f"llmstack {__version__}")
        typer.echo(f"Python {sys.version.split()[0]} on {platform.platform()}")
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
def quickstart(
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model to use"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    skip_pull: bool = typer.Option(False, "--skip-pull", help="Skip model pull check"),
) -> None:
    """Zero-to-running in one command: check deps, pull model, create config."""
    from llmstack.cli.commands.quickstart import quickstart as _quickstart
    _quickstart(model=model, ollama_url=ollama_url, skip_pull=skip_pull)


@app.command()
def init(
    preset: str = typer.Option(None, "--preset", "-p", help="Preset: chat, rag, agent, secure"),
    directory: str = typer.Option(None, "--dir", "-d", help="Target directory"),
) -> None:
    """Create a new llmstack.yaml configuration file."""
    from pathlib import Path
    from llmstack.cli.commands.init import init as _init
    _init(preset=preset, directory=Path(directory) if directory else None)


@app.command(name="config")
def config_cmd(
    action: str = typer.Argument("show", help="Action: show, validate, path"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format: yaml, json"),
) -> None:
    """Inspect and validate llmstack.yaml configuration."""
    from llmstack.cli.commands.config import config_validate, config_show, config_path

    actions = {
        "validate": config_validate,
        "show": lambda: config_show(output_format=format),
        "path": config_path,
    }
    handler = actions.get(action)
    if handler:
        handler()
    else:
        from llmstack.cli.console import console
        console.print(f"[error]Unknown action: {action}[/]")
        console.print(f"Available: {', '.join(actions.keys())}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
) -> None:
    """Start the gateway API server directly (no Docker required)."""
    from llmstack.cli.commands.serve import serve as _serve
    _serve(host=host, port=port, reload=reload, workers=workers)


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
def profile(
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model to profile"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    runs: int = typer.Option(4, "--runs", "-n", help="Number of test prompts"),
) -> None:
    """Quick performance profile: tokens/sec, latency per prompt."""
    from llmstack.cli.commands.profile import profile as _profile
    _profile(model=model, ollama_url=ollama_url, runs=runs)


@app.command()
def compare(
    prompt: str = typer.Argument(..., help="Prompt to send to all models"),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated model names"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
) -> None:
    """Compare outputs from multiple models side-by-side."""
    from llmstack.cli.commands.compare import compare as _compare
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    _compare(prompt=prompt, models=model_list, ollama_url=ollama_url)


@app.command()
def export(
    output: str = typer.Option("docker-compose.yml", "--output", "-o", help="Output file path"),
) -> None:
    """Export llmstack.yaml as a standalone docker-compose.yml."""
    from llmstack.cli.commands.export import export as _export
    _export(output=output)


@app.command()
def traces(
    gateway_url: str = typer.Option(None, "--gateway-url", "-g", help="Gateway URL"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of traces to show"),
    model: str = typer.Option(None, "--model", "-m", help="Filter by model name"),
) -> None:
    """View recent request traces with latency, cost, and quality scores."""
    from llmstack.cli.commands.traces import traces as _traces
    _traces(gateway_url=gateway_url, limit=limit, model_filter=model)


@app.command()
def cost(
    gateway_url: str = typer.Option(None, "--gateway-url", "-g", help="Gateway URL"),
) -> None:
    """Show cost, usage, and savings summary from the gateway."""
    from llmstack.cli.commands.cost import cost as _cost
    _cost(gateway_url=gateway_url)


@app.command()
def playground(
    gateway_url: str = typer.Option(None, "--url", "-u", help="Gateway URL"),
) -> None:
    """Open the LLMStack Web UI playground in your browser."""
    from llmstack.cli.commands.playground import playground as _playground
    _playground(gateway_url=gateway_url)


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model name to pull (e.g. llama3.2, mistral, codellama)"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
) -> None:
    """Pull a model from the Ollama registry with progress display."""
    from llmstack.cli.commands.pull import pull as _pull
    _pull(model=model, ollama_url=ollama_url)


@app.command(name="models")
def models_cmd(
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    gateway_url: str = typer.Option(None, "--gateway-url", "-g", help="Running gateway URL"),
) -> None:
    """List all available models from Ollama and the gateway."""
    from llmstack.cli.commands.models import models as _models
    _models(ollama_url=ollama_url, gateway_url=gateway_url)


@app.command()
def info() -> None:
    """Show detailed system, hardware, and project information."""
    from llmstack.cli.commands.info import info as _info
    _info()


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
    repo: list[str] = typer.Option(None, "--repo", "-r", help="Additional repo paths for multi-repo support"),
) -> None:
    """Ask questions about local files using a local LLM."""
    from llmstack.cli.commands.ask import ask as _ask
    # Merge extra repo paths into files list
    all_files = list(files or [])
    for r in (repo or []):
        all_files.append(Path(r))
    _ask(
        question=question,
        files=all_files if all_files else files,
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


@app.command()
def review(
    target: str = typer.Argument("", help="Git ref or range (e.g. HEAD~2..HEAD, branch..main)"),
    pr: str = typer.Option(None, "--pr", help="GitHub PR URL to fetch and review"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option("terminal", "--output", "-o", help="Output format: terminal, markdown, json"),
    output_file: str = typer.Option(None, "--output-file", "-f", help="Save report to file"),
    severity: str = typer.Option(None, "--severity", "-s", help="Filter by severity: CRITICAL, WARNING, INFO"),
    staged: bool = typer.Option(False, "--staged", help="Review staged changes only"),
    commits: int = typer.Option(1, "--commits", "-c", help="Number of recent commits to review"),
) -> None:
    """AI-powered code review for git diffs and GitHub PRs."""
    from llmstack.cli.commands.review import review as _review
    _review(
        target=target, pr_url=pr, model=model, ollama_url=ollama_url,
        output_format=output, severity=severity, output_file=output_file,
        staged=staged, commits=commits,
    )


@app.command()
def fix(
    description: str = typer.Argument("", help="Description of the issue to fix"),
    file: str = typer.Option(None, "--file", "-f", help="File to fix"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show patch without applying"),
    no_interactive: bool = typer.Option(False, "--no-interactive", help="Apply patch without confirmation"),
) -> None:
    """AI-powered auto-fix: generate and apply a patch for a code issue."""
    from llmstack.cli.commands.fix import fix as _fix
    _fix(
        description=description, file=file, model=model, ollama_url=ollama_url,
        dry_run=dry_run, interactive=not no_interactive,
    )


@app.command()
def docs(
    target: str = typer.Argument(None, help="File or directory to document"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    doc_type: str = typer.Option("docstrings", "--type", "-t", help="Type: docstrings, readme"),
    write: bool = typer.Option(False, "--write", "-w", help="Write changes to files"),
) -> None:
    """Generate documentation, docstrings, or README using AI."""
    from llmstack.cli.commands.docs import docs as _docs
    _docs(target=target, output=output, model=model, ollama_url=ollama_url,
          doc_type=doc_type, write=write)


@app.command(name="test")
def test_cmd(
    target: str = typer.Argument(None, help="File or directory to generate tests for"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    framework: str = typer.Option("pytest", "--framework", "-F", help="Test framework: pytest, jest"),
    write: bool = typer.Option(False, "--write", "-w", help="Write test files"),
    coverage: bool = typer.Option(False, "--coverage", help="Include coverage hints"),
) -> None:
    """Generate AI-powered test cases for your code."""
    from llmstack.cli.commands.test_gen import test_gen as _test_gen
    _test_gen(target=target, output=output, model=model, ollama_url=ollama_url,
              framework=framework, write=write, coverage=coverage)


@app.command(name="diff")
def diff_cmd(
    target: str = typer.Argument("HEAD~1..HEAD", help="Git range or ref to explain"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    staged: bool = typer.Option(False, "--staged", help="Explain staged changes"),
    commits: int = typer.Option(1, "--commits", "-c", help="Number of recent commits"),
    file: str = typer.Option(None, "--file", "-f", help="Limit diff to specific file"),
) -> None:
    """Explain a git diff in plain English."""
    from llmstack.cli.commands.diff_explain import diff_explain as _diff
    _diff(target=target, model=model, ollama_url=ollama_url,
          staged=staged, commits=commits, file=file)


@app.command()
def watch(
    directory: str = typer.Argument(".", help="Directory to watch"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    patterns: str = typer.Option("*.py,*.js,*.ts", "--patterns", "-p", help="File patterns to watch (comma-separated)"),
    debounce: float = typer.Option(2.0, "--debounce", "-d", help="Debounce seconds between analyses"),
) -> None:
    """Watch files for changes and get real-time AI suggestions."""
    from llmstack.cli.commands.watch import watch as _watch
    _watch(directory=directory, model=model, ollama_url=ollama_url,
           patterns=patterns, debounce=debounce)


@app.command()
def commit(
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    push: bool = typer.Option(False, "--push", help="Push after committing"),
    all_changes: bool = typer.Option(False, "--all", "-a", help="Stage all changes before generating message"),
) -> None:
    """Generate a conventional commit message with AI and optionally apply it."""
    from llmstack.cli.commands.commit_gen import commit_gen as _commit_gen
    _commit_gen(model=model, ollama_url=ollama_url, push=push, all_changes=all_changes)


@app.command()
def security(
    target: str = typer.Argument(None, help="File or directory to audit"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option("terminal", "--output", "-o", help="Output format: terminal, markdown, json"),
    output_file: str = typer.Option(None, "--output-file", "-f", help="Save report to file"),
    severity: str = typer.Option(None, "--severity", "-s", help="Filter by severity: CRITICAL, HIGH, MEDIUM, LOW"),
) -> None:
    """AI-powered security audit with OWASP Top 10 and CWE references."""
    from llmstack.cli.commands.security import security as _security
    _security(target=target, model=model, ollama_url=ollama_url,
              output_format=output, output_file=output_file, severity=severity)


@app.command(name="history")
def history_cmd(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of conversations to show"),
    search: str = typer.Option(None, "--search", "-s", help="Search query"),
    index_dir: str = typer.Option(None, "--index-dir", help="Custom index directory"),
) -> None:
    """View and search your ask conversation history."""
    from llmstack.cli.commands.history import history as _history
    _history(index_dir=index_dir, limit=limit, search=search)


@app.command(name="export-conv")
def export_conv_cmd(
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("markdown", "--format", "-f", help="Format: markdown, json"),
    index_dir: str = typer.Option(None, "--index-dir", help="Custom index directory"),
) -> None:
    """Export conversation history from the persistent index."""
    from llmstack.cli.commands.export_conv import export_conv as _export_conv
    _export_conv(output=output, format=format, index_dir=index_dir)


# --- Adaptive Learning Pipeline commands ---

@app.command(name="learn")
def learn_cmd(
    action: str = typer.Argument("status", help="Action: status, train, rollback, feedback, export, reset, preferences, patterns, versions"),
    limit: int = typer.Option(20, "--limit", "-n", help="Limit results"),
    output: str = typer.Option(None, "--output", "-o", help="Output path for export"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Export format: jsonl, json, hf, backup"),
    feedback_type: str = typer.Option(None, "--type", "-t", help="Filter feedback by type"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm destructive actions"),
    force: bool = typer.Option(False, "--force", help="Force action without checks"),
) -> None:
    """Adaptive learning pipeline — your AI gets smarter over time.

    Actions:
      status      Show pipeline status and metrics
      train       Trigger a training run
      rollback    Rollback to previous model version
      feedback    Show collected feedback
      export      Export learning data
      reset       Reset all learning data
      preferences Show learned user preferences
      patterns    Show learned code patterns
      versions    Show model version history
    """
    from llmstack.cli.commands.learn import (
        learn_status,
        learn_train,
        learn_rollback,
        learn_feedback,
        learn_export,
        learn_reset,
        learn_preferences,
        learn_patterns,
        learn_versions,
    )

    actions = {
        "status": lambda: learn_status(),
        "train": lambda: learn_train(force=force),
        "rollback": lambda: learn_rollback(),
        "feedback": lambda: learn_feedback(limit=limit, feedback_type=feedback_type),
        "export": lambda: learn_export(output=output, format=format),
        "reset": lambda: learn_reset(confirm=confirm),
        "preferences": lambda: learn_preferences(),
        "patterns": lambda: learn_patterns(),
        "versions": lambda: learn_versions(),
    }

    handler = actions.get(action)
    if handler:
        handler()
    else:
        from llmstack.cli.console import console
        console.print(f"[red]Unknown action: {action}[/]")
        console.print(f"Available: {', '.join(actions.keys())}")


@app.command(name="analytics")
def analytics_cmd(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
    output: str = typer.Option(None, "--output", "-o", help="Export analytics to JSON"),
) -> None:
    """View usage statistics, trends, and performance metrics."""
    from llmstack.cli.commands.analytics import analytics as _analytics
    _analytics(days=days, output=output)


@app.command(name="workflow")
def workflow_cmd(
    action: str = typer.Argument("list", help="Action: list, show, run, create, delete"),
    name: str = typer.Argument("", help="Workflow name"),
    steps: str = typer.Option(None, "--steps", help="JSON steps for create"),
    description: str = typer.Option("", "--desc", "-d", help="Workflow description"),
) -> None:
    """Run automated command pipelines — chain multiple llmstack commands."""
    from llmstack.cli.commands.workflow import (
        workflow_list, workflow_show, workflow_run,
        workflow_create, workflow_delete,
    )

    actions = {
        "list": lambda: workflow_list(),
        "show": lambda: workflow_show(name),
        "run": lambda: workflow_run(name),
        "create": lambda: workflow_create(name, steps or "[]", description),
        "delete": lambda: workflow_delete(name),
    }
    handler = actions.get(action)
    if handler:
        handler()
    else:
        console.print(f"[error]Unknown action: {action}[/]")


@app.command(name="dead-code")
def dead_code_cmd(
    target: str = typer.Argument(None, help="Directory to scan"),
    confidence: str = typer.Option(None, "--confidence", "-c", help="Filter: high, medium, low"),
    code_type: str = typer.Option(None, "--type", "-t", help="Filter: function, class, import"),
    output: str = typer.Option(None, "--output", "-o", help="Save report to JSON"),
) -> None:
    """Find unused functions, classes, and imports in your codebase."""
    from llmstack.cli.commands.dead_code import dead_code as _dead_code
    _dead_code(target=target, confidence=confidence, code_type=code_type, output=output)


@app.command(name="complexity")
def complexity_cmd(
    target: str = typer.Argument(None, help="File or directory to analyze"),
    threshold: int = typer.Option(10, "--threshold", "-t", help="Cyclomatic complexity threshold"),
    sort_by: str = typer.Option("complexity", "--sort", "-s", help="Sort by: complexity, cognitive, lines, name"),
    output: str = typer.Option(None, "--output", "-o", help="Save report to JSON"),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all functions, not just complex ones"),
) -> None:
    """Analyze code complexity — cyclomatic, cognitive, and maintainability index."""
    from llmstack.cli.commands.complexity import complexity as _complexity
    _complexity(target=target, threshold=threshold, sort_by=sort_by,
                output=output, show_all=show_all)


@app.command(name="hooks")
def hooks_cmd(
    action: str = typer.Argument("list", help="Action: list, install, install-all, remove, show"),
    hook_name: str = typer.Argument(None, help="Hook name: pre-commit, commit-msg, pre-push, post-checkout"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing hooks"),
) -> None:
    """Set up AI-powered git hooks for automated code quality."""
    from llmstack.cli.commands.hooks import hooks as _hooks
    _hooks(action=action, hook_name=hook_name, force=force)


@app.command()
def scaffold(
    description: str = typer.Argument("", help="Project description"),
    preset: str = typer.Option(None, "--preset", "-p", help="Preset: fastapi, nextjs, react, cli-python, express, etc."),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show structure without creating files"),
) -> None:
    """Generate a complete project scaffold from description or preset."""
    from llmstack.cli.commands.scaffold import scaffold as _scaffold
    _scaffold(description=description, preset=preset, model=model,
              ollama_url=ollama_url, output_dir=output_dir, dry_run=dry_run)


@app.command(name="tokens")
def tokens_cmd(
    target: str = typer.Argument(None, help="File or directory to analyze"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model for context window reference"),
    no_recursive: bool = typer.Option(False, "--no-recursive", help="Don't recurse into subdirectories"),
    no_files: bool = typer.Option(False, "--no-files", help="Only show summary"),
) -> None:
    """Count tokens in files and check if they fit in the model's context window."""
    from llmstack.cli.commands.tokens import tokens as _tokens
    _tokens(target=target, model=model, recursive=not no_recursive, show_files=not no_files)


@app.command(name="prompt")
def prompt_cmd(
    action: str = typer.Argument("list", help="Action: list, show, use, create, delete"),
    name: str = typer.Argument("", help="Template name"),
    var: list[str] = typer.Option(None, "--var", "-v", help="Variable: key=value"),
    template: str = typer.Option(None, "--template", "-t", help="Template text for create"),
    description: str = typer.Option("", "--desc", "-d", help="Template description"),
    category: str = typer.Option("custom", "--category", "-c", help="Template category"),
) -> None:
    """Manage reusable prompt templates — 12 built-in + custom templates."""
    from llmstack.cli.commands.prompt import (
        prompt_list, prompt_show, prompt_use, prompt_create, prompt_delete,
    )

    actions = {
        "list": lambda: prompt_list(category=category if category != "custom" else None),
        "show": lambda: prompt_show(name=name),
        "use": lambda: prompt_use(name=name, variables=var),
        "create": lambda: prompt_create(name=name, template=template or "", description=description, category=category),
        "delete": lambda: prompt_delete(name=name),
    }
    handler = actions.get(action)
    if handler:
        handler()
    else:
        console.print(f"[error]Unknown action: {action}[/]")


@app.command()
def diagram(
    target: str = typer.Argument(None, help="File or directory to diagram"),
    diagram_type: str = typer.Option("architecture", "--type", "-t", help="Type: architecture, class, sequence, flow, er, dependency, state"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option(None, "--output", "-o", help="Save diagram to file (.md or .mmd)"),
) -> None:
    """Generate Mermaid architecture diagrams from code using AI."""
    from llmstack.cli.commands.diagram import diagram as _diagram
    _diagram(target=target, diagram_type=diagram_type, model=model,
             ollama_url=ollama_url, output=output)


@app.command(name="deps")
def deps_cmd(
    target: str = typer.Argument(None, help="Project directory to analyze"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model for AI analysis"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option(None, "--output", "-o", help="Save report to JSON"),
    no_security: bool = typer.Option(False, "--no-security", help="Skip AI security analysis"),
    no_updates: bool = typer.Option(False, "--no-updates", help="Skip update check"),
) -> None:
    """Analyze project dependencies — security, updates, and licensing."""
    from llmstack.cli.commands.deps import deps as _deps
    _deps(target=target, check_updates=not no_updates, check_security=not no_security,
          model=model, ollama_url=ollama_url, output=output)


@app.command(name="snippet")
def snippet_cmd(
    action: str = typer.Argument("list", help="Action: save, search, show, delete, tags, export, stats"),
    query: str = typer.Argument("", help="Search query or snippet ID"),
    file: str = typer.Option(None, "--file", "-f", help="File to save as snippet"),
    title: str = typer.Option(None, "--title", "-t", help="Snippet title"),
    tags: str = typer.Option(None, "--tags", help="Comma-separated tags"),
    description: str = typer.Option("", "--desc", "-d", help="Snippet description"),
    lines: str = typer.Option(None, "--lines", "-l", help="Line range (e.g., 10-20)"),
    language: str = typer.Option(None, "--language", help="Filter by language"),
    output: str = typer.Option(None, "--output", "-o", help="Export output file"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Manage your code snippet library — save, search, and reuse code."""
    from llmstack.cli.commands.snippet import (
        snippet_save, snippet_search, snippet_show,
        snippet_delete, snippet_tags, snippet_export, snippet_stats,
    )

    actions = {
        "save": lambda: snippet_save(file=file, title=title, tags=tags, description=description, lines=lines),
        "search": lambda: snippet_search(query=query, language=language, limit=limit),
        "list": lambda: snippet_search(query="", language=language, limit=limit),
        "show": lambda: snippet_show(snippet_id=query),
        "delete": lambda: snippet_delete(snippet_id=query),
        "tags": lambda: snippet_tags(),
        "export": lambda: snippet_export(output=output),
        "stats": lambda: snippet_stats(),
    }
    handler = actions.get(action)
    if handler:
        handler()
    else:
        console.print(f"[error]Unknown action: {action}[/]")
        console.print(f"Available: {', '.join(actions.keys())}")


@app.command()
def refactor(
    target: str = typer.Argument(..., help="File to analyze for refactoring"),
    strategy: str = typer.Option("clean", "--strategy", "-s", help="Strategy: clean, performance, type-safety, solid, testability"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option(None, "--output", "-o", help="Save report to JSON file"),
    apply: bool = typer.Option(False, "--apply", help="Apply suggested refactoring"),
) -> None:
    """AI-powered refactoring suggestions with multiple strategies."""
    from llmstack.cli.commands.refactor import refactor as _refactor
    _refactor(target=target, strategy=strategy, model=model, ollama_url=ollama_url,
              output=output, apply=apply)


@app.command()
def explain(
    target: str = typer.Argument(..., help="File to explain"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="Specific function/class to explain"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    level: str = typer.Option("mid", "--level", "-l", help="Explanation level: beginner, mid, senior"),
    output: str = typer.Option(None, "--output", "-o", help="Save explanation to file"),
) -> None:
    """Explain code in detail with diagrams and examples."""
    from llmstack.cli.commands.explain import explain as _explain
    _explain(target=target, symbol=symbol, model=model, ollama_url=ollama_url,
             level=level, output=output)


@app.command()
def changelog(
    since: str = typer.Option(None, "--since", "-s", help="Git ref to start from (tag, commit, branch)"),
    version: str = typer.Option(None, "--version", "-v", help="Version label for the changelog"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option(None, "--output", "-o", help="Save changelog to file"),
    max_commits: int = typer.Option(100, "--max-commits", help="Maximum commits to include"),
) -> None:
    """Auto-generate a changelog from git history using AI."""
    from llmstack.cli.commands.changelog import changelog as _changelog
    _changelog(since=since, version=version, model=model, ollama_url=ollama_url,
               output=output, max_commits=max_commits)


@app.command()
def translate(
    file: str = typer.Argument(..., help="Source file to translate"),
    to_lang: str = typer.Argument(..., help="Target language (python, javascript, go, rust, java, etc.)"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model name"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama API URL"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    write: bool = typer.Option(False, "--write", "-w", help="Write translated code to file"),
) -> None:
    """Translate code between programming languages using AI."""
    from llmstack.cli.commands.translate import translate as _translate
    _translate(file=file, to_lang=to_lang, model=model, ollama_url=ollama_url,
               output=output, write=write)


@app.command(name="backup")
def backup_cmd(
    output: str = typer.Option(None, "--output", "-o", help="Output archive path"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Data directory to backup"),
    include_models: bool = typer.Option(False, "--include-models", help="Include model files"),
) -> None:
    """Create a backup of all LLMStack configuration and data."""
    from llmstack.cli.commands.backup import backup as _backup
    _backup(output=output, data_dir=data_dir, include_models=include_models)


@app.command(name="restore")
def restore_cmd(
    archive: str = typer.Argument(..., help="Path to backup archive (.tar.gz)"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Target data directory"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
) -> None:
    """Restore LLMStack configuration and data from a backup."""
    from llmstack.cli.commands.backup import restore as _restore
    _restore(archive=archive, data_dir=data_dir, force=force)


@app.command(name="backups")
def list_backups_cmd(
    directory: str = typer.Option(".", "--dir", "-d", help="Directory to search"),
) -> None:
    """List available backup files."""
    from llmstack.cli.commands.backup import list_backups as _list
    _list(directory=directory)
