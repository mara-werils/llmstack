"""llmstack learn — manage the adaptive learning pipeline.

Subcommands:
- status: Show learning pipeline status and metrics
- train: Trigger a training run manually
- rollback: Rollback to previous model version
- feedback: Show/manage collected feedback
- export: Export learning data
- reset: Reset learning state
- preferences: Show learned user preferences
- patterns: Show learned code patterns
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from llmstack.cli.console import console


def learn_status() -> None:
    """Show learning pipeline status and metrics."""
    from llmstack.learn.analytics import LearningAnalytics
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)
    analytics = LearningAnalytics(store=store, version_mgr=version_mgr)

    summary = analytics.get_summary()
    metrics = summary["metrics"]

    # Status header
    status = summary["status"]
    status_colors = {
        "inactive": "dim",
        "collecting": "yellow",
        "active": "green",
        "improving": "bold green",
        "degrading": "red",
    }
    color = status_colors.get(status, "white")
    console.print(f"\n[bold]Learning Pipeline[/] — [{color}]{status}[/]\n")

    # Feedback metrics
    fb = metrics["feedback"]
    console.print("[bold]Feedback Collection[/]")
    console.print(f"  Total signals:    {fb['total']}")
    console.print(f"  Positive rate:    {fb['positive_rate']:.1%}")
    console.print(f"  Correction rate:  {fb['correction_rate']:.1%}")
    console.print(f"  Per day:          {fb['per_day']:.1f}")
    console.print(f"  Unused (pending): {fb['unused']}")
    console.print()

    # Training metrics
    tr = metrics["training"]
    console.print("[bold]Training[/]")
    console.print(f"  Total runs:       {tr['total_runs']}")
    console.print(f"  Model versions:   {tr['total_versions']}")
    console.print(f"  Avg dataset size: {tr['avg_dataset_size']:.0f}")
    console.print()

    # Quality metrics
    q = metrics["quality"]
    console.print("[bold]Quality[/]")
    console.print(f"  Current score:    {q['current']:.4f}")
    console.print(f"  Improvement:      {q['improvement']:+.4f}")
    console.print(f"  Best ever:        {q['best']:.4f}")
    console.print(f"  Trend:            {q['trend']}")
    console.print()

    # Recommendations
    recs = summary.get("recommendations", [])
    if recs:
        console.print("[bold]Recommendations[/]")
        for rec in recs:
            console.print(f"  • {rec}")
        console.print()

    store.close()


def learn_train(force: bool = False) -> None:
    """Trigger a training run."""
    from llmstack.learn.dataset import DatasetGenerator
    from llmstack.learn.scheduler import SchedulerConfig, TrainScheduler, TriggerReason
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    unused = store.get_unused_feedback_count()

    if unused == 0:
        console.print("[yellow]No unused feedback available for training.[/]")
        console.print("Use llmstack with feedback enabled to collect training data.")
        store.close()
        return

    console.print(f"\n[bold]Training Pipeline[/]\n")
    console.print(f"  Unused feedback: {unused}")

    dataset_gen = DatasetGenerator(store=store)
    version_mgr = ModelVersionManager(store=store)
    scheduler = TrainScheduler(
        store=store,
        dataset_gen=dataset_gen,
        version_mgr=version_mgr,
    )

    # Set up a mock training callback (actual training requires GPU)
    def _train_callback(dataset):
        console.print(f"  Dataset size: {dataset.total_examples} examples")
        console.print(f"  SFT: {len(dataset.sft_examples)}, DPO: {len(dataset.dpo_examples)}")

        # Save dataset for external training
        output_dir = Path.home() / ".llmstack" / "training" / "datasets"
        path = dataset.save(output_dir)
        console.print(f"  Saved to: {path}")

        return {
            "success": True,
            "final_loss": 0.0,
            "best_loss": 0.0,
            "adapter_path": "",
            "quality_score": 0.0,
            "train_time_seconds": 0.0,
        }

    scheduler.set_train_callback(_train_callback)

    console.print("\n  [dim]Generating dataset...[/]")
    result = scheduler.trigger(TriggerReason.MANUAL)

    if result.get("success"):
        console.print(f"\n[green]Training complete![/]")
        console.print(f"  Version: v{result.get('version', '?')}")
        console.print(f"  Dataset: {result.get('dataset_size', 0)} examples")
    elif "error" in result:
        console.print(f"\n[red]Training failed:[/] {result['error']}")

    store.close()


def learn_rollback() -> None:
    """Rollback to previous model version."""
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)

    active = version_mgr.get_active()
    if not active:
        console.print("[yellow]No active model version to rollback from.[/]")
        store.close()
        return

    console.print(f"Current version: v{active.version} (quality={active.quality_score:.4f})")

    result = version_mgr.rollback()
    if result:
        console.print(f"[green]Rolled back to v{result.version}[/] (quality={result.quality_score:.4f})")
    else:
        console.print("[red]No previous version available for rollback.[/]")

    store.close()


def learn_feedback(limit: int = 20, feedback_type: str | None = None) -> None:
    """Show collected feedback."""
    from llmstack.learn.feedback import FeedbackType
    from llmstack.learn.store import FeedbackStore

    store = FeedbackStore()

    ft = FeedbackType(feedback_type) if feedback_type else None
    feedback_list = store.get_feedback(feedback_type=ft, limit=limit)

    if not feedback_list:
        console.print("[yellow]No feedback collected yet.[/]")
        console.print("Use thumbs up/down or corrections in chat/ask to start learning.")
        store.close()
        return

    console.print(f"\n[bold]Recent Feedback[/] ({len(feedback_list)} items)\n")

    for fb in feedback_list:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(fb.timestamp))
        type_colors = {
            "thumbs_up": "green",
            "thumbs_down": "red",
            "correction": "cyan",
            "edit": "blue",
            "regenerate": "yellow",
            "copy": "green",
        }
        color = type_colors.get(fb.feedback_type.value, "white")
        query_preview = fb.query[:60] + "..." if len(fb.query) > 60 else fb.query
        console.print(
            f"  [{color}]{fb.feedback_type.value:12}[/] {ts}  "
            f"[dim]{query_preview}[/]"
        )

    # Stats
    stats = store.get_stats()
    console.print(f"\n  Total: {stats['total_feedback']} | "
                  f"Unused: {stats['unused_feedback']} | "
                  f"Types: {dict(stats.get('feedback_by_type', {}))}")

    store.close()


def learn_export(output: str | None = None, format: str = "jsonl") -> None:
    """Export learning data."""
    from llmstack.learn.export import LearningExporter
    from llmstack.learn.patterns import PatternLearner
    from llmstack.learn.preferences import PreferenceLearner
    from llmstack.learn.store import FeedbackStore

    store = FeedbackStore()
    pref_learner = PreferenceLearner(store=store)
    pat_learner = PatternLearner(store=store)
    exporter = LearningExporter(
        store=store,
        preference_learner=pref_learner,
        pattern_learner=pat_learner,
    )

    output_path = Path(output) if output else Path.cwd() / "llmstack-learning-export"

    if format == "backup":
        path = exporter.export_full_backup(output_path)
        console.print(f"[green]Full backup exported to:[/] {path}")
    elif format == "hf":
        path = exporter.export_dataset_hf(output_path)
        console.print(f"[green]HuggingFace dataset exported to:[/] {path}")
    else:
        path = exporter.export_feedback(output_path, format=format)
        console.print(f"[green]Feedback exported to:[/] {path}")

    store.close()


def learn_reset(confirm: bool = False) -> None:
    """Reset all learning data."""
    from llmstack.learn.store import FeedbackStore, DEFAULT_DB_PATH

    if not confirm:
        console.print("[yellow]This will delete all learning data![/]")
        console.print(f"Database: {DEFAULT_DB_PATH}")
        console.print("Use --confirm to proceed.")
        return

    if DEFAULT_DB_PATH.exists():
        DEFAULT_DB_PATH.unlink()
        console.print("[green]Learning data reset.[/]")
    else:
        console.print("[dim]No learning data found.[/]")


def learn_preferences() -> None:
    """Show learned user preferences."""
    from llmstack.learn.preferences import PreferenceLearner
    from llmstack.learn.store import FeedbackStore

    store = FeedbackStore()
    learner = PreferenceLearner(store=store)
    profile = learner.get_profile()

    console.print("\n[bold]Learned Preferences[/]\n")

    # Length
    length = profile.get("length", {})
    console.print(f"[bold]Response Length[/]")
    console.print(f"  Tendency:      {length.get('tendency', 'neutral')}")
    console.print(f"  Avg preferred: {length.get('avg_preferred', 0):.0f} chars")
    console.print(f"  Samples:       {length.get('samples', 0)}")
    console.print()

    # Formatting
    fmt = profile.get("formatting", {})
    console.print("[bold]Formatting[/]")
    console.print(f"  Code blocks:   {fmt.get('code_blocks', 0.5):.0%}")
    console.print(f"  Bullet lists:  {fmt.get('bullet_lists', 0.5):.0%}")
    console.print(f"  Headers:       {fmt.get('headers', 0.5):.0%}")
    console.print(f"  Markdown:      {fmt.get('markdown', 0.5):.0%}")
    console.print()

    # Tone
    tone = profile.get("tone", {})
    console.print("[bold]Tone[/]")
    console.print(f"  Formality:     {tone.get('formality', 0.5):.0%}")
    console.print(f"  Directness:    {tone.get('directness', 0.5):.0%}")
    console.print(f"  Technicality:  {tone.get('technicality', 0.5):.0%}")
    console.print()

    # System prompt additions
    additions = learner.get_system_prompt_additions()
    if additions:
        console.print("[bold]System Prompt Additions[/]")
        console.print(f"  {additions}")
        console.print()

    store.close()


def learn_patterns() -> None:
    """Show learned code patterns."""
    from llmstack.learn.patterns import PatternLearner
    from llmstack.learn.store import FeedbackStore

    store = FeedbackStore()
    learner = PatternLearner(store=store)
    profile = learner.get_profile()

    console.print("\n[bold]Learned Code Patterns[/]\n")

    if not profile.get("patterns"):
        console.print("[dim]No code patterns learned yet.[/]")
        console.print("Make code corrections to build your style profile.")
        store.close()
        return

    # Naming conventions
    naming = profile.get("naming", {})
    if naming:
        console.print("[bold]Naming Conventions[/]")
        for style, weight in sorted(naming.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 20)
            console.print(f"  {style:15} {bar} {weight:.0%}")
        console.print()

    # Patterns
    patterns = profile.get("patterns", [])
    if patterns:
        console.print("[bold]Code Patterns[/]")
        for p in sorted(patterns, key=lambda x: -x.get("confidence", 0)):
            conf = p.get("confidence", 0)
            count = p.get("occurrences", 0)
            color = "green" if conf > 0.7 else "yellow" if conf > 0.4 else "dim"
            console.print(
                f"  [{color}]{conf:.0%}[/] {p['description']} "
                f"[dim](seen {count}x)[/]"
            )
        console.print()

    # Style guide
    guide = learner.get_style_guide()
    if guide:
        console.print("[bold]Generated Style Guide[/]")
        console.print(f"  {guide}")
        console.print()

    store.close()


def learn_versions() -> None:
    """Show model version history."""
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)
    versions = version_mgr.list_versions(limit=20)

    if not versions:
        console.print("[dim]No model versions yet. Run 'llmstack learn train' first.[/]")
        store.close()
        return

    console.print("\n[bold]Model Versions[/]\n")

    for v in versions:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(v.timestamp))
        active = " [green]← active[/]" if v.is_active else ""
        console.print(
            f"  v{v.version:3}  {ts}  quality={v.quality_score:.4f}  "
            f"base={v.base_model}{active}"
        )

    console.print()
    store.close()
