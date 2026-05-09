"""CLI command: llmstack finetune — one-command fine-tuning pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llmstack.cli.console import console


def finetune(
    data: str,
    base_model: str = "unsloth/llama-3.2-1b-instruct-bnb-4bit",
    method: str = "qlora",
    output: str = "./finetune-output",
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    lora_r: int | None = None,
    max_seq_length: int = 2048,
    eval_split: float = 0.1,
    export_gguf: bool = False,
    export_ollama: str | None = None,
    quantization: str = "q4_k_m",
    system_prompt: str = "",
    resume: str | None = None,
) -> None:
    """Run the complete fine-tuning pipeline."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    from llmstack.finetune.data import DatasetConfig, prepare_dataset
    from llmstack.finetune.hyperparams import auto_hyperparams, estimate_model_size
    from llmstack.finetune.trainer import TrainConfig, Trainer

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[error]Data file not found: {data}[/]")
        sys.exit(1)

    output_dir = Path(output)

    # ──── Step 1: Prepare data ────
    console.print()
    console.print("[bold cyan]Step 1/3[/] Preparing dataset...")

    ds_config = DatasetConfig(
        system_prompt=system_prompt,
        eval_split=eval_split,
    )

    train_data, eval_data, stats = prepare_dataset(
        data_path, config=ds_config, output_dir=output_dir / "data",
    )

    if not train_data:
        console.print("[error]No training examples found in dataset.[/]")
        sys.exit(1)

    # Show dataset stats
    data_table = Table(title="Dataset", show_header=False, border_style="cyan")
    data_table.add_column("Key", style="bold")
    data_table.add_column("Value")
    data_table.add_row("Format", stats.source_format)
    data_table.add_row("Total examples", str(stats.total_examples))
    data_table.add_row("Train / Eval", f"{stats.train_examples} / {stats.eval_examples}")
    data_table.add_row("Skipped", str(stats.skipped))
    data_table.add_row("Avg tokens (in/out)", f"{stats.avg_input_tokens} / {stats.avg_output_tokens}")
    data_table.add_row("Total tokens", f"{stats.total_tokens:,}")
    console.print(data_table)
    console.print()

    # ──── Step 2: Configure and train ────
    console.print("[bold cyan]Step 2/3[/] Training...")

    model_size = estimate_model_size(base_model)
    hp = auto_hyperparams(
        num_examples=stats.train_examples,
        model_size_b=model_size,
        method=method,
        max_seq_length=max_seq_length,
    )

    # Apply user overrides
    if epochs is not None:
        hp.epochs = epochs
    if lr is not None:
        hp.learning_rate = lr
    if batch_size is not None:
        hp.batch_size = batch_size
    if lora_r is not None:
        hp.lora_r = lora_r
        hp.lora_alpha = lora_r * 2

    # Show training config
    config_table = Table(title="Training Config", show_header=False, border_style="cyan")
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Base model", base_model)
    config_table.add_row("Method", method.upper())
    config_table.add_row("LoRA rank / alpha", f"{hp.lora_r} / {hp.lora_alpha}")
    config_table.add_row("Epochs", str(hp.epochs))
    config_table.add_row("Batch size", f"{hp.batch_size} (accum: {hp.gradient_accumulation_steps})")
    config_table.add_row("Learning rate", f"{hp.learning_rate:.1e}")
    config_table.add_row("Max seq length", str(hp.max_seq_length))
    config_table.add_row("Scheduler", hp.lr_scheduler)
    console.print(config_table)
    console.print()

    train_config = TrainConfig(
        base_model=base_model,
        method=method,
        output_dir=str(output_dir),
        hyperparams=hp,
        resume_from=resume,
    )

    trainer = Trainer(train_config)

    # Progress tracking
    loss_values: list[float] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("loss: {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=100, loss=0.0)

        def on_progress(step: int, total: int, loss: float) -> None:
            loss_values.append(loss)
            pct = (step / max(total, 1)) * 100
            progress.update(task, completed=pct, loss=loss)

        trainer.set_progress_callback(on_progress)
        result = trainer.train(train_data, eval_data)

    console.print()

    if not result.success:
        console.print(Panel(
            f"[error]Training failed:[/]\n{result.error}",
            title="Error", border_style="red",
        ))
        sys.exit(1)

    # Show results
    results_table = Table(title="Training Results", show_header=False, border_style="green")
    results_table.add_column("Key", style="bold")
    results_table.add_column("Value")
    results_table.add_row("Final loss", f"{result.final_loss:.4f}")
    results_table.add_row("Best loss", f"{result.best_loss:.4f}")
    results_table.add_row("Total steps", str(result.total_steps))
    results_table.add_row("Training time", f"{result.train_time_seconds:.0f}s")
    results_table.add_row("Adapter path", result.adapter_path)
    console.print(results_table)

    # Show loss curve (ASCII sparkline)
    if loss_values:
        _print_loss_curve(loss_values)

    # ──── Step 3: Export ────
    if export_gguf or export_ollama:
        console.print()
        console.print("[bold cyan]Step 3/3[/] Exporting...")

        from llmstack.finetune.export import export_gguf as do_export, create_ollama_model

        if export_gguf or export_ollama:
            gguf_result = do_export(
                adapter_path=result.adapter_path,
                base_model=base_model,
                output_path=str(output_dir / "model.gguf"),
                quantization=quantization,
            )

            if gguf_result.success:
                console.print(f"  [success]GGUF exported: {gguf_result.gguf_path} ({gguf_result.size_mb:.0f} MB)[/]")

                if export_ollama:
                    ollama_result = create_ollama_model(
                        gguf_path=gguf_result.gguf_path,
                        model_name=export_ollama,
                        system_prompt=system_prompt,
                    )
                    if ollama_result.success:
                        console.print(f"  [success]Ollama model created: {ollama_result.ollama_model}[/]")
                        console.print(f"  Run: [info]ollama run {ollama_result.ollama_model}[/]")
                    else:
                        console.print(f"  [error]Ollama export failed: {ollama_result.error}[/]")
            else:
                console.print(f"  [error]GGUF export failed: {gguf_result.error}[/]")
    else:
        console.print("\n  [info]Tip: add --export-gguf or --export-ollama NAME to create a deployable model[/]")

    # Save full result
    result_path = output_dir / "train_result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2))
    console.print(f"\n[success]Done! Results saved to {output_dir}[/]")


def _print_loss_curve(values: list[float]) -> None:
    """Print an ASCII loss curve using block characters."""
    if len(values) < 2:
        return

    # Downsample to fit in terminal
    width = min(60, len(values))
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]

    min_val = min(sampled)
    max_val = max(sampled)
    span = max_val - min_val
    if span < 1e-8:
        return

    bars = " ▁▂▃▄▅▆▇█"

    # Normalize to bar height
    normalized = [(v - min_val) / span for v in sampled]
    bar_chars = []
    for v in normalized:
        # Invert: higher loss = taller bar at top
        idx = int((1.0 - v) * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        bar_chars.append(bars[idx])

    "".join(bar_chars)
    console.print(f"\n  Loss curve: {max_val:.4f} {''.join(bar_chars)} {min_val:.4f}")
    console.print(f"  {'':>14}{'start':}<{len(bar_chars) - 3}s{'end'}")
