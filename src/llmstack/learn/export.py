"""Export utilities — export learned data in portable formats.

Supports exporting:
- Feedback as JSONL (for external training pipelines)
- Datasets in HuggingFace format
- Preferences as JSON config
- Code patterns as style guide
- Full learning state backup/restore
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

from llmstack.learn.store import FeedbackStore
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.patterns import PatternLearner

logger = logging.getLogger(__name__)


class LearningExporter:
    """Export learned data in various portable formats."""

    def __init__(
        self,
        store: FeedbackStore,
        preference_learner: PreferenceLearner | None = None,
        pattern_learner: PatternLearner | None = None,
    ):
        self.store = store
        self.preference_learner = preference_learner
        self.pattern_learner = pattern_learner

    def export_feedback(
        self,
        output_path: Path,
        format: str = "jsonl",
        limit: int = 10000,
    ) -> Path:
        """Export feedback as JSONL file."""
        feedback = self.store.get_feedback(limit=limit)

        output_path = output_path.with_suffix(f".{format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for fb in feedback:
                    f.write(json.dumps(fb.to_dict()) + "\n")
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump([fb.to_dict() for fb in feedback], f, indent=2)

        logger.info("Exported %d feedback items to %s", len(feedback), output_path)
        return output_path

    def export_dataset_hf(
        self,
        output_dir: Path,
        split_ratio: float = 0.9,
    ) -> Path:
        """Export as HuggingFace-compatible dataset (train/test split)."""
        from llmstack.learn.dataset import DatasetGenerator, DatasetStrategy

        gen = DatasetGenerator(self.store)
        dataset = gen.generate(strategy=DatasetStrategy.MIXED, max_examples=10000)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Split into train/test
        sft = dataset.sft_examples
        split_idx = int(len(sft) * split_ratio)
        train_examples = sft[:split_idx]
        test_examples = sft[split_idx:]

        # Write train split
        train_path = output_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Write test split
        test_path = output_dir / "test.jsonl"
        with open(test_path, "w") as f:
            for ex in test_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Write dataset card
        card_path = output_dir / "README.md"
        card_path.write_text(
            f"# LLMStack Learning Dataset\n\n"
            f"Auto-generated from user feedback.\n\n"
            f"- Train examples: {len(train_examples)}\n"
            f"- Test examples: {len(test_examples)}\n"
            f"- Generated: {time.strftime('%Y-%m-%d %H:%M')}\n"
        )

        logger.info(
            "Exported HF dataset: %d train, %d test",
            len(train_examples),
            len(test_examples),
        )
        return output_dir

    def export_preferences(self, output_path: Path) -> Path:
        """Export learned preferences as JSON."""
        output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {"exported_at": time.time()}

        if self.preference_learner:
            data["preferences"] = self.preference_learner.get_profile()
            data["system_prompt_additions"] = self.preference_learner.get_system_prompt_additions()

        if self.pattern_learner:
            data["code_patterns"] = self.pattern_learner.get_profile()
            data["style_guide"] = self.pattern_learner.get_style_guide()

        output_path.write_text(json.dumps(data, indent=2))
        logger.info("Exported preferences to %s", output_path)
        return output_path

    def export_full_backup(self, output_path: Path) -> Path:
        """Create a full backup of all learning data."""
        output_path = output_path.with_suffix(".tar.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp directory for backup contents
        tmp_dir = output_path.parent / f"_backup_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Export feedback
            self.export_feedback(tmp_dir / "feedback", format="jsonl")

            # Export preferences
            if self.preference_learner:
                self.export_preferences(tmp_dir / "preferences")

            # Export patterns
            if self.pattern_learner:
                patterns_path = tmp_dir / "code_patterns.json"
                patterns_path.write_text(json.dumps(self.pattern_learner.get_profile(), indent=2))

            # Export stats
            stats_path = tmp_dir / "stats.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "exported_at": time.time(),
                        "stats": self.store.get_stats(),
                    },
                    indent=2,
                )
            )

            # Copy database file
            db_path = Path(self.store.db_path)
            if db_path.exists():
                shutil.copy2(db_path, tmp_dir / "learning.db")

            # Create tarball
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(str(tmp_dir), arcname="llmstack-learning-backup")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info("Full backup created at %s", output_path)
        return output_path

    def import_backup(self, backup_path: Path) -> dict[str, Any]:
        """Import a learning data backup."""
        if not backup_path.exists():
            return {"error": f"Backup not found: {backup_path}"}

        tmp_dir = backup_path.parent / f"_import_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(str(tmp_dir))

            # Find the backup directory
            backup_dir = tmp_dir / "llmstack-learning-backup"
            if not backup_dir.exists():
                # Try finding any subdirectory
                dirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
                backup_dir = dirs[0] if dirs else tmp_dir

            # Import feedback
            feedback_path = backup_dir / "feedback.jsonl"
            imported = 0
            if feedback_path.exists():
                from llmstack.learn.feedback import Feedback

                with open(feedback_path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            fb = Feedback.from_dict(data)
                            self.store.add_feedback(fb)
                            imported += 1

            return {
                "success": True,
                "imported_feedback": imported,
            }

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
