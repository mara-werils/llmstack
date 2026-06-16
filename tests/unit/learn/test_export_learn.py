"""Tests for exporting learned data in portable formats."""

from __future__ import annotations

import json
import tarfile

import pytest

from llmstack.learn.export import LearningExporter
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.patterns import PatternLearner
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def store_with_feedback(store):
    """Store seeded with a variety of feedback types."""
    for i in range(6):
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query=f"How do I do task {i}?",
                response=f"You can do it like this: bad_answer_{i}",
                correction=f"The correct way is: good_answer_{i} with much more detail",
                model="llama3.2",
            )
        )
    for i in range(4):
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query=f"Explain concept {i}",
                response=f"This is a great explanation of concept {i} with details",
                model="llama3.2",
            )
        )
    return store


@pytest.fixture
def preference_learner(store, tmp_path):
    learner = PreferenceLearner(store=store, preferences_path=tmp_path / "prefs.json")
    for _ in range(8):
        learner.learn_from_feedback(
            Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="How do I print in Python?",
                response="To print in Python you use the print() function with a long winded "
                "explanation that goes on and on with unnecessary verbosity.",
                correction="Use `print('hi')`",
            )
        )
    return learner


@pytest.fixture
def pattern_learner(store, tmp_path):
    learner = PatternLearner(store=store, patterns_path=tmp_path / "patterns.json")
    for _ in range(5):
        learner.learn_from_correction(
            original="def f(x):\n    return x",
            correction="```python\ndef compute(value):\n    return value\n```",
        )
    return learner


class TestProperties:
    def test_has_learners_false_when_absent(self, store):
        exporter = LearningExporter(store=store)
        assert exporter.has_preference_learner is False
        assert exporter.has_pattern_learner is False

    def test_has_learners_true_when_present(self, store, preference_learner, pattern_learner):
        exporter = LearningExporter(
            store=store,
            preference_learner=preference_learner,
            pattern_learner=pattern_learner,
        )
        assert exporter.has_preference_learner is True
        assert exporter.has_pattern_learner is True


class TestExportFeedback:
    def test_export_jsonl(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out = exporter.export_feedback(tmp_path / "fb")

        assert out == tmp_path / "fb.jsonl"
        assert out.exists()

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 10
        # Each line is valid JSON with expected keys
        first = json.loads(lines[0])
        assert "feedback_type" in first
        assert "query" in first

    def test_export_json(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out = exporter.export_feedback(tmp_path / "fb", format="json")

        assert out == tmp_path / "fb.json"
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 10

    def test_export_creates_parent_dirs(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out = exporter.export_feedback(tmp_path / "nested" / "deep" / "fb")
        assert out.exists()
        assert out.parent.is_dir()

    def test_export_unknown_format_writes_no_content(self, store_with_feedback, tmp_path):
        """Unknown format branch falls through; file is not created."""
        exporter = LearningExporter(store=store_with_feedback)
        out = exporter.export_feedback(tmp_path / "fb", format="csv")
        assert out == tmp_path / "fb.csv"
        # Neither jsonl nor json branch ran, so nothing is written.
        assert not out.exists()

    def test_export_empty_store(self, store, tmp_path):
        exporter = LearningExporter(store=store)
        out = exporter.export_feedback(tmp_path / "fb")
        assert out.exists()
        assert out.read_text() == ""

    def test_export_respects_limit(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out = exporter.export_feedback(tmp_path / "fb", limit=3)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3


class TestExportDatasetHF:
    def test_export_dataset(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out_dir = exporter.export_dataset_hf(tmp_path / "hf")

        assert out_dir == tmp_path / "hf"
        assert (out_dir / "train.jsonl").exists()
        assert (out_dir / "test.jsonl").exists()

        card = (out_dir / "README.md").read_text()
        assert "LLMStack Learning Dataset" in card
        assert "Train examples:" in card
        assert "Test examples:" in card

    def test_export_dataset_split_ratio(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out_dir = exporter.export_dataset_hf(tmp_path / "hf", split_ratio=0.5)

        train_lines = (out_dir / "train.jsonl").read_text().strip()
        # README must reflect the produced split counts.
        card = (out_dir / "README.md").read_text()
        assert "Train examples:" in card
        if train_lines:
            for line in train_lines.splitlines():
                json.loads(line)  # valid jsonl

    def test_export_dataset_empty_store(self, store, tmp_path):
        exporter = LearningExporter(store=store)
        out_dir = exporter.export_dataset_hf(tmp_path / "hf")
        assert (out_dir / "train.jsonl").read_text() == ""
        assert (out_dir / "test.jsonl").read_text() == ""


class TestExportPreferences:
    def test_export_with_both_learners(
        self, store, preference_learner, pattern_learner, tmp_path
    ):
        exporter = LearningExporter(
            store=store,
            preference_learner=preference_learner,
            pattern_learner=pattern_learner,
        )
        out = exporter.export_preferences(tmp_path / "prefs")

        assert out == tmp_path / "prefs.json"
        data = json.loads(out.read_text())
        assert "exported_at" in data
        assert "preferences" in data
        assert "system_prompt_additions" in data
        assert "code_patterns" in data
        assert "style_guide" in data

    def test_export_without_learners(self, store, tmp_path):
        exporter = LearningExporter(store=store)
        out = exporter.export_preferences(tmp_path / "prefs")
        data = json.loads(out.read_text())
        assert "exported_at" in data
        assert "preferences" not in data
        assert "code_patterns" not in data

    def test_export_preferences_only(self, store, preference_learner, tmp_path):
        exporter = LearningExporter(store=store, preference_learner=preference_learner)
        out = exporter.export_preferences(tmp_path / "prefs")
        data = json.loads(out.read_text())
        assert "preferences" in data
        assert "code_patterns" not in data


class TestFullBackup:
    def test_full_backup_creates_tarball(
        self, store_with_feedback, preference_learner, pattern_learner, tmp_path
    ):
        exporter = LearningExporter(
            store=store_with_feedback,
            preference_learner=preference_learner,
            pattern_learner=pattern_learner,
        )
        out = exporter.export_full_backup(tmp_path / "backup")

        assert out == tmp_path / "backup.tar.gz"
        assert out.exists()
        assert tarfile.is_tarfile(out)

        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        assert any("feedback.jsonl" in n for n in names)
        assert any("preferences.json" in n for n in names)
        assert any("code_patterns.json" in n for n in names)
        assert any("stats.json" in n for n in names)
        assert any("learning.db" in n for n in names)

    def test_full_backup_cleans_temp_dir(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        exporter.export_full_backup(tmp_path / "backup")
        # No leftover _backup_* temp directories.
        leftovers = list(tmp_path.glob("_backup_*"))
        assert leftovers == []

    def test_full_backup_without_learners(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        out = exporter.export_full_backup(tmp_path / "backup")
        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        # Feedback and stats always present; learner files absent.
        assert any("feedback.jsonl" in n for n in names)
        assert any("stats.json" in n for n in names)
        assert not any("preferences.json" in n for n in names)
        assert not any("code_patterns.json" in n for n in names)


class TestImportBackup:
    def test_import_round_trip(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        backup = exporter.export_full_backup(tmp_path / "backup")

        # Fresh store to import into.
        target_store = FeedbackStore(db_path=tmp_path / "target.db")
        try:
            target_exporter = LearningExporter(store=target_store)
            result = target_exporter.import_backup(backup)
            assert result["success"] is True
            assert result["imported_feedback"] == 10
            # Feedback actually landed in the target store.
            assert len(target_store.get_feedback(limit=100)) == 10
        finally:
            target_store.close()

    def test_import_missing_backup(self, store, tmp_path):
        exporter = LearningExporter(store=store)
        result = exporter.import_backup(tmp_path / "does_not_exist.tar.gz")
        assert "error" in result
        assert "Backup not found" in result["error"]

    def test_import_fallback_directory(self, store, tmp_path):
        """Backup whose inner dir is not the canonical name still imports."""
        # Build a backup with a non-standard top-level directory name.
        src = tmp_path / "weirdname"
        src.mkdir()
        fb = Feedback(
            feedback_type=FeedbackType.THUMBS_UP,
            query="hello there",
            response="general kenobi",
        )
        (src / "feedback.jsonl").write_text(json.dumps(fb.to_dict()) + "\n")

        archive = tmp_path / "weird.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(str(src), arcname="weirdname")

        target = FeedbackStore(db_path=tmp_path / "t.db")
        try:
            exporter = LearningExporter(store=target)
            result = exporter.import_backup(archive)
            assert result["success"] is True
            assert result["imported_feedback"] == 1
        finally:
            target.close()

    def test_import_no_feedback_file(self, store, tmp_path):
        """Backup tarball with no feedback.jsonl imports zero items."""
        src = tmp_path / "llmstack-learning-backup"
        src.mkdir()
        (src / "stats.json").write_text("{}")

        archive = tmp_path / "nofb.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(str(src), arcname="llmstack-learning-backup")

        exporter = LearningExporter(store=store)
        result = exporter.import_backup(archive)
        assert result["success"] is True
        assert result["imported_feedback"] == 0

    def test_import_skips_blank_lines(self, store, tmp_path):
        """Blank lines in feedback.jsonl are ignored."""
        src = tmp_path / "llmstack-learning-backup"
        src.mkdir()
        fb = Feedback(
            feedback_type=FeedbackType.THUMBS_UP,
            query="q one",
            response="r one",
        )
        (src / "feedback.jsonl").write_text(
            json.dumps(fb.to_dict()) + "\n\n   \n" + json.dumps(fb.to_dict()) + "\n"
        )

        archive = tmp_path / "blanks.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(str(src), arcname="llmstack-learning-backup")

        target = FeedbackStore(db_path=tmp_path / "t.db")
        try:
            exporter = LearningExporter(store=target)
            result = exporter.import_backup(archive)
            assert result["imported_feedback"] == 2
        finally:
            target.close()

    def test_import_cleans_temp_dir(self, store_with_feedback, tmp_path):
        exporter = LearningExporter(store=store_with_feedback)
        backup = exporter.export_full_backup(tmp_path / "backup")
        target = FeedbackStore(db_path=tmp_path / "t.db")
        try:
            LearningExporter(store=target).import_backup(backup)
        finally:
            target.close()
        assert list(tmp_path.glob("_import_*")) == []
