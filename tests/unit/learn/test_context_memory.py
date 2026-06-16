"""Tests for context memory — learning effective context for query types."""

from __future__ import annotations

import json

import pytest

from llmstack.learn.context_memory import (
    ContextMemory,
    ContextProfile,
    ContextSignal,
)
from llmstack.learn.feedback import FeedbackType
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def memory(store, tmp_path):
    return ContextMemory(store=store, memory_path=tmp_path / "context_memory.json")


class TestContextSignal:
    def test_effectiveness_no_uses(self):
        sig = ContextSignal(context_type="file", query_pattern="howto")
        # Neutral default when never used.
        assert sig.effectiveness == 0.5

    def test_effectiveness_with_uses(self):
        sig = ContextSignal(
            context_type="file",
            query_pattern="howto",
            positive_count=3,
            negative_count=1,
            total_uses=4,
        )
        assert sig.effectiveness == pytest.approx(0.75)

    def test_effectiveness_all_negative(self):
        sig = ContextSignal(
            context_type="git",
            query_pattern="debugging",
            positive_count=0,
            total_uses=5,
        )
        assert sig.effectiveness == 0.0


class TestContextProfile:
    def test_get_best_contexts_orders_by_effectiveness(self):
        profile = ContextProfile()
        profile.signals = {
            "howto:file": ContextSignal(
                context_type="file",
                query_pattern="howto",
                positive_count=4,
                total_uses=5,
            ),
            "howto:git": ContextSignal(
                context_type="git",
                query_pattern="howto",
                positive_count=1,
                total_uses=5,
            ),
            "howto:docs": ContextSignal(
                context_type="docs",
                query_pattern="howto",
                positive_count=5,
                total_uses=5,
            ),
        }
        best = profile.get_best_contexts("howto")
        assert best == ["docs", "file", "git"]

    def test_get_best_contexts_respects_top_k(self):
        profile = ContextProfile()
        profile.signals = {
            f"howto:c{i}": ContextSignal(
                context_type=f"c{i}",
                query_pattern="howto",
                positive_count=i,
                total_uses=5,
            )
            for i in range(5)
        }
        best = profile.get_best_contexts("howto", top_k=2)
        assert len(best) == 2

    def test_get_best_contexts_ignores_low_use_signals(self):
        profile = ContextProfile()
        profile.signals = {
            "howto:file": ContextSignal(
                context_type="file",
                query_pattern="howto",
                positive_count=2,
                total_uses=2,  # below min threshold of 3
            ),
        }
        assert profile.get_best_contexts("howto") == []

    def test_get_best_contexts_filters_by_pattern(self):
        profile = ContextProfile()
        profile.signals = {
            "howto:file": ContextSignal(
                context_type="file",
                query_pattern="howto",
                positive_count=3,
                total_uses=3,
            ),
            "debugging:git": ContextSignal(
                context_type="git",
                query_pattern="debugging",
                positive_count=3,
                total_uses=3,
            ),
        }
        assert profile.get_best_contexts("debugging") == ["git"]

    def test_to_dict_round_trips_fields(self):
        profile = ContextProfile()
        profile.last_updated = 123.0
        profile.query_patterns = {"howto": ["file", "git"]}
        profile.signals = {
            "howto:file": ContextSignal(
                context_type="file",
                query_pattern="howto",
                positive_count=2,
                negative_count=1,
                total_uses=4,
            ),
        }
        d = profile.to_dict()
        assert d["last_updated"] == 123.0
        assert d["query_patterns"] == {"howto": ["file", "git"]}
        sig = d["signals"]["howto:file"]
        assert sig["context_type"] == "file"
        assert sig["query_pattern"] == "howto"
        assert sig["positive_count"] == 2
        assert sig["negative_count"] == 1
        assert sig["total_uses"] == 4
        assert sig["effectiveness"] == pytest.approx(0.5, abs=0.01)


class TestContextMemory:
    def test_initial_state(self, memory):
        assert memory.signal_count == 0
        assert memory.tracked_patterns == []
        assert memory.recommend_context("anything") == []

    def test_record_creates_signals(self, memory):
        memory.record_context_use("How do I run tests?", ["file", "docs"])
        assert memory.signal_count == 2
        assert "howto" in memory.tracked_patterns

    def test_record_positive_feedback(self, memory):
        memory.record_context_use(
            "How do I run tests?",
            ["file"],
            feedback_type=FeedbackType.THUMBS_UP,
        )
        sig = memory.profile.signals["howto:file"]
        assert sig.positive_count == 1
        assert sig.negative_count == 0
        assert sig.total_uses == 1

    def test_record_copy_counts_positive(self, memory):
        memory.record_context_use(
            "How do I run tests?",
            ["file"],
            feedback_type=FeedbackType.COPY,
        )
        assert memory.profile.signals["howto:file"].positive_count == 1

    def test_record_negative_feedback(self, memory):
        memory.record_context_use(
            "Fix this broken parser",
            ["git"],
            feedback_type=FeedbackType.THUMBS_DOWN,
        )
        sig = memory.profile.signals["debugging:git"]
        assert sig.negative_count == 1
        assert sig.positive_count == 0

    def test_record_regenerate_counts_negative(self, memory):
        memory.record_context_use(
            "Fix this broken parser",
            ["git"],
            feedback_type=FeedbackType.REGENERATE,
        )
        assert memory.profile.signals["debugging:git"].negative_count == 1

    def test_record_neutral_feedback_type_only_counts_use(self, memory):
        # ABANDON is neither positive nor negative.
        memory.record_context_use(
            "How do I run tests?",
            ["file"],
            feedback_type=FeedbackType.ABANDON,
        )
        sig = memory.profile.signals["howto:file"]
        assert sig.total_uses == 1
        assert sig.positive_count == 0
        assert sig.negative_count == 0

    def test_record_without_feedback(self, memory):
        memory.record_context_use("How do I run tests?", ["file"])
        sig = memory.profile.signals["howto:file"]
        assert sig.total_uses == 1
        assert sig.positive_count == 0

    def test_record_accumulates_uses(self, memory):
        for _ in range(3):
            memory.record_context_use(
                "How do I run tests?",
                ["file"],
                feedback_type=FeedbackType.THUMBS_UP,
            )
        sig = memory.profile.signals["howto:file"]
        assert sig.total_uses == 3
        assert sig.positive_count == 3

    def test_record_updates_query_patterns(self, memory):
        memory.record_context_use("How do I run tests?", ["file", "docs"])
        assert set(memory.profile.query_patterns["howto"]) == {"file", "docs"}
        # Recording more context types extends the pattern set.
        memory.record_context_use("How do I run tests?", ["git"])
        assert "git" in memory.profile.query_patterns["howto"]

    def test_query_patterns_capped_at_ten(self, memory):
        many = [f"ctx{i}" for i in range(20)]
        memory.record_context_use("How do I run tests?", many)
        assert len(memory.profile.query_patterns["howto"]) <= 10

    def test_last_updated_set(self, memory):
        assert memory.profile.last_updated == 0.0
        memory.record_context_use("How do I run tests?", ["file"])
        assert memory.profile.last_updated > 0.0

    def test_recommend_context(self, memory):
        for _ in range(4):
            memory.record_context_use(
                "How do I run tests?",
                ["docs"],
                feedback_type=FeedbackType.THUMBS_UP,
            )
        for _ in range(4):
            memory.record_context_use(
                "How do I run tests?",
                ["git"],
                feedback_type=FeedbackType.THUMBS_DOWN,
            )
        recs = memory.recommend_context("How do I install this?")
        # docs (all positive) should rank above git (all negative).
        assert recs[0] == "docs"

    def test_effectiveness_report(self, memory):
        for _ in range(3):
            memory.record_context_use(
                "How do I run tests?",
                ["docs"],
                feedback_type=FeedbackType.THUMBS_UP,
            )
        report = memory.get_effectiveness_report()
        assert "howto" in report
        assert report["howto"]["docs"] == pytest.approx(1.0)

    def test_effectiveness_report_excludes_low_use(self, memory):
        memory.record_context_use(
            "How do I run tests?",
            ["docs"],
            feedback_type=FeedbackType.THUMBS_UP,
        )
        # Only 1 use, below the >=3 threshold.
        assert memory.get_effectiveness_report() == {}


class TestClassifyQuery:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("How do I run tests?", "howto"),
            ("how can i build this", "howto"),
            ("Why is this slow?", "explanation"),
            ("Explain the architecture", "explanation"),
            ("What is a closure", "explanation"),
            ("Fix the bug in parser", "debugging"),
            ("This is broken", "debugging"),
            ("not working as expected", "debugging"),
            ("Write a function to sort", "generation"),
            ("create a new module", "generation"),
            ("implement caching", "generation"),
            ("Review this PR", "review"),
            ("optimize the loop", "review"),
            ("refactor the handler", "review"),
            ("add test coverage", "generation"),
            ("write a spec for this", "generation"),
            ("Random unrelated sentence", "general"),
        ],
    )
    def test_classify(self, memory, query, expected):
        assert memory._classify_query(query) == expected


class TestPersistence:
    def test_save_writes_file(self, store, tmp_path):
        path = tmp_path / "context_memory.json"
        memory = ContextMemory(store=store, memory_path=path)
        memory.record_context_use("How do I run tests?", ["file"])
        assert path.exists()
        data = json.loads(path.read_text())
        assert "howto:file" in data["signals"]

    def test_save_creates_parent_dirs(self, store, tmp_path):
        path = tmp_path / "nested" / "dir" / "context_memory.json"
        memory = ContextMemory(store=store, memory_path=path)
        memory.record_context_use("How do I run tests?", ["file"])
        assert path.exists()

    def test_load_round_trip(self, store, tmp_path):
        path = tmp_path / "context_memory.json"
        m1 = ContextMemory(store=store, memory_path=path)
        for _ in range(3):
            m1.record_context_use(
                "How do I run tests?",
                ["file"],
                feedback_type=FeedbackType.THUMBS_UP,
            )

        m2 = ContextMemory(store=store, memory_path=path)
        assert m2.signal_count == 1
        sig = m2.profile.signals["howto:file"]
        assert sig.total_uses == 3
        assert sig.positive_count == 3
        assert m2.profile.query_patterns["howto"] == ["file"]

    def test_load_missing_file_returns_empty_profile(self, store, tmp_path):
        path = tmp_path / "does_not_exist.json"
        memory = ContextMemory(store=store, memory_path=path)
        assert memory.signal_count == 0
        assert memory.profile.last_updated == 0.0

    def test_load_corrupt_json_returns_empty_profile(self, store, tmp_path):
        path = tmp_path / "context_memory.json"
        path.write_text("{not valid json")
        memory = ContextMemory(store=store, memory_path=path)
        assert memory.signal_count == 0

    def test_load_missing_required_key_returns_empty_profile(self, store, tmp_path):
        path = tmp_path / "context_memory.json"
        # Signal entry missing the required "context_type" key triggers KeyError.
        path.write_text(
            json.dumps(
                {
                    "signals": {"howto:file": {"query_pattern": "howto"}},
                    "query_patterns": {},
                    "last_updated": 0,
                }
            )
        )
        memory = ContextMemory(store=store, memory_path=path)
        assert memory.signal_count == 0

    def test_load_uses_defaults_for_optional_counts(self, store, tmp_path):
        path = tmp_path / "context_memory.json"
        path.write_text(
            json.dumps(
                {
                    "signals": {
                        "howto:file": {
                            "context_type": "file",
                            "query_pattern": "howto",
                        }
                    },
                    "query_patterns": {"howto": ["file"]},
                    "last_updated": 42.0,
                }
            )
        )
        memory = ContextMemory(store=store, memory_path=path)
        sig = memory.profile.signals["howto:file"]
        assert sig.positive_count == 0
        assert sig.negative_count == 0
        assert sig.total_uses == 0
        assert memory.profile.last_updated == 42.0
