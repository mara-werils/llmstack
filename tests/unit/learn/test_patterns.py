"""Tests for code pattern learning."""

from __future__ import annotations

import json

import pytest

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.patterns import (
    CodePattern,
    CodeStyleProfile,
    PatternLearner,
)
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def learner(store, tmp_path):
    return PatternLearner(
        store=store,
        patterns_path=tmp_path / "patterns.json",
    )


class TestCodePattern:
    def test_to_dict_truncates_examples(self):
        p = CodePattern(
            name="sorted_imports",
            description="Keep imports sorted.",
            examples=[f"ex{i}" for i in range(10)],
            counter_examples=[f"ce{i}" for i in range(10)],
            confidence=0.123456,
            occurrences=7,
        )
        d = p.to_dict()
        assert d["name"] == "sorted_imports"
        assert d["description"] == "Keep imports sorted."
        assert len(d["examples"]) == 5
        assert len(d["counter_examples"]) == 3
        assert d["confidence"] == 0.123  # rounded to 3 places
        assert d["occurrences"] == 7


class TestCodeStyleProfile:
    def test_to_dict_roundtrip_shape(self):
        prof = CodeStyleProfile(
            naming={"snake_case": 0.9},
            patterns=[CodePattern(name="p", description="d")],
            language_preferences={"python": True},
            last_updated=123.0,
            total_code_corrections=4,
        )
        d = prof.to_dict()
        assert d["naming"] == {"snake_case": 0.9}
        assert d["patterns"][0]["name"] == "p"
        assert d["language_preferences"] == {"python": True}
        assert d["last_updated"] == 123.0
        assert d["total_code_corrections"] == 4

    def test_to_style_guide_empty(self):
        assert CodeStyleProfile().to_style_guide() == ""

    def test_to_style_guide_naming_above_threshold(self):
        prof = CodeStyleProfile(naming={"snake_case": 0.8, "camelCase": 0.2})
        guide = prof.to_style_guide()
        assert "snake_case naming convention" in guide

    def test_to_style_guide_naming_below_threshold(self):
        # Dominant style only 0.55 -> no naming line emitted.
        prof = CodeStyleProfile(naming={"snake_case": 0.55, "camelCase": 0.45})
        assert prof.to_style_guide() == ""

    def test_to_style_guide_high_confidence_pattern(self):
        prof = CodeStyleProfile(
            patterns=[
                CodePattern(name="p1", description="Do thing one.", confidence=0.9),
                CodePattern(name="p2", description="Low conf.", confidence=0.5),
            ]
        )
        guide = prof.to_style_guide()
        assert "Do thing one." in guide
        assert "Low conf." not in guide


class TestPatternLearnerProperties:
    def test_initial_counts(self, learner):
        assert learner.pattern_count == 0
        assert learner.high_confidence_patterns == []

    def test_pattern_count_and_high_confidence(self, learner):
        learner.profile.patterns.append(
            CodePattern(name="a", description="d", confidence=0.9)
        )
        learner.profile.patterns.append(
            CodePattern(name="b", description="d", confidence=0.3)
        )
        assert learner.pattern_count == 2
        hc = learner.high_confidence_patterns
        assert len(hc) == 1
        assert hc[0].name == "a"


class TestLearnFromCorrection:
    def test_non_code_content_ignored(self, learner):
        learner.learn_from_correction("hello world", "goodbye world")
        assert learner.profile.total_code_corrections == 0
        assert learner.pattern_count == 0

    def test_code_content_increments_counter(self, learner):
        learner.learn_from_correction(
            "def foo(): pass", "def foo():\n    return 1"
        )
        assert learner.profile.total_code_corrections == 1
        assert learner.profile.last_updated > 0

    def test_learn_naming_snake_case(self, learner):
        # Correction introduces snake_case identifiers.
        for _ in range(20):
            learner.learn_from_correction(
                "def fooBar():\n    myValue = 1",
                "def foo_bar():\n    my_value = 1\n    other_thing = 2",
            )
        assert "snake_case" in learner.profile.naming
        assert learner.profile.naming["snake_case"] > 0.5

    def test_learn_naming_no_identifiers_in_correction(self, learner):
        # Correction has code indicators (indentation) but no extractable
        # identifiers, so _learn_naming returns early.
        before = dict(learner.profile.naming)
        learner.learn_from_correction("def x(): pass", "    # just a comment\n")
        assert learner.profile.naming == before

    def test_learn_formatting_trailing_newline(self, learner):
        learner.learn_from_correction("def f(): pass", "def f(): pass\n")
        names = {p.name for p in learner.profile.patterns}
        assert "trailing_newline" in names

    def test_learn_formatting_blank_lines(self, learner):
        original = "def a(): pass\ndef b(): pass"
        correction = "def a(): pass\n\n\ndef b(): pass"
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "blank_lines" in names

    def test_learn_formatting_line_length(self, learner):
        long_line = "x = " + "a" * 120
        original = "\n".join([long_line] * 3) + "\ndef f(): pass"
        correction = "x = a\ny = b\nz = c\ndef f(): pass"
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "line_length" in names

    def test_learn_error_handling_try_except(self, learner):
        original = "def f():\n    risky()"
        correction = "def f():\n    try:\n        risky()\n    except Exception:\n        pass"
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "error_handling" in names

    def test_learn_error_handling_specific_exceptions(self, learner):
        original = "try:\n    x()\nexcept:\n    pass"
        correction = "try:\n    x()\nexcept ValueError:\n    pass"
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "specific_exceptions" in names

    def test_learn_imports_sorted(self, learner):
        original = "import os\nimport abc\ndef f(): pass"
        correction = "import abc\nimport os\ndef f(): pass"
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "sorted_imports" in names

    def test_learn_imports_no_imports_in_correction(self, learner):
        # Correction has code but no import lines -> _learn_imports returns early.
        original = "import os\ndef f(): pass"
        correction = "def f():\n    return 1"
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "sorted_imports" not in names
        assert "from_imports" not in names

    def test_learn_imports_from_preference(self, learner):
        original = "def f(): pass"
        correction = (
            "from a import x\n"
            "from b import y\n"
            "from c import z\n"
            "def f(): pass"
        )
        learner.learn_from_correction(original, correction)
        names = {p.name for p in learner.profile.patterns}
        assert "from_imports" in names


class TestUpdatePattern:
    def test_update_existing_pattern_increments(self, learner):
        for _ in range(5):
            learner.learn_from_correction("def f(): pass", "def f(): pass\n")
        matching = [p for p in learner.profile.patterns if p.name == "trailing_newline"]
        assert len(matching) == 1
        pat = matching[0]
        assert pat.occurrences == 5
        assert pat.confidence == pytest.approx(0.5)  # 5/10

    def test_confidence_caps_at_one(self, learner):
        for _ in range(20):
            learner.learn_from_correction("def f(): pass", "def f(): pass\n")
        pat = next(p for p in learner.profile.patterns if p.name == "trailing_newline")
        assert pat.confidence == 1.0

    def test_examples_capped_at_five(self, learner):
        # trailing_newline stores correction[-20:] as example; vary the tail
        # so each correction produces a distinct example string.
        for i in range(8):
            original = "def f(): pass"
            correction = f"def f(): pass # tag{i}\n"
            learner.learn_from_correction(original, correction)
        pat = next(p for p in learner.profile.patterns if p.name == "trailing_newline")
        assert len(pat.examples) <= 5


class TestClassifyNaming:
    @pytest.mark.parametrize(
        "identifier,expected",
        [
            ("my_value", "snake_case"),
            ("myValue", "camelCase"),
            ("MyClass", "PascalCase"),
            # All-caps with an underscore is matched by the PascalCase branch
            # first (it appears before the UPPER_SNAKE branch in the source).
            ("MAX_SIZE", "PascalCase"),
            ("value", None),
            ("data", None),
        ],
    )
    def test_classify(self, learner, identifier, expected):
        assert learner._classify_naming(identifier) == expected


class TestLearnFromFeedback:
    def test_correction_feedback_learns(self, learner):
        fb = Feedback(
            feedback_type=FeedbackType.CORRECTION,
            response="def fooBar(): pass",
            correction="def foo_bar():\n    return 1",
        )
        learner.learn_from_feedback(fb)
        assert learner.profile.total_code_corrections == 1

    def test_edit_feedback_learns(self, learner):
        fb = Feedback(
            feedback_type=FeedbackType.EDIT,
            response="def f(): pass",
            correction="def f(): pass\n",
        )
        learner.learn_from_feedback(fb)
        assert learner.profile.total_code_corrections == 1

    def test_non_correction_feedback_ignored(self, learner):
        fb = Feedback(
            feedback_type=FeedbackType.THUMBS_UP,
            response="def f(): pass",
            correction="def f(): pass\n",
        )
        learner.learn_from_feedback(fb)
        assert learner.profile.total_code_corrections == 0

    def test_correction_without_text_ignored(self, learner):
        fb = Feedback(
            feedback_type=FeedbackType.CORRECTION,
            response="def f(): pass",
            correction="",
        )
        learner.learn_from_feedback(fb)
        assert learner.profile.total_code_corrections == 0


class TestRebuildFromHistory:
    def test_rebuild_from_store(self, store, learner):
        for _ in range(3):
            store.add_feedback(
                Feedback(
                    feedback_type=FeedbackType.CORRECTION,
                    response="def fooBar(): pass",
                    correction="def foo_bar():\n    return 1",
                )
            )
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.EDIT,
                response="def g(): pass",
                correction="def g(): pass\n",
            )
        )
        learner.rebuild_from_history(limit=100)
        # 3 corrections + 1 edit, all code content.
        assert learner.profile.total_code_corrections == 4

    def test_rebuild_resets_existing_profile(self, learner, store):
        learner.profile.total_code_corrections = 999
        learner.profile.patterns.append(CodePattern(name="stale", description="d"))
        learner.rebuild_from_history()
        # No feedback in store -> fresh empty profile.
        assert learner.profile.total_code_corrections == 0
        assert learner.pattern_count == 0


class TestStyleGuideAndProfileGetters:
    def test_get_style_guide(self, learner):
        learner.profile.naming = {"snake_case": 0.9}
        assert "snake_case" in learner.get_style_guide()

    def test_get_profile(self, learner):
        learner.profile.total_code_corrections = 2
        prof = learner.get_profile()
        assert prof["total_code_corrections"] == 2
        assert "patterns" in prof
        assert "naming" in prof


class TestPersistence:
    def test_save_creates_file(self, learner):
        learner.learn_from_correction("def f(): pass", "def f(): pass\n")
        assert learner.patterns_path.exists()
        data = json.loads(learner.patterns_path.read_text())
        assert "patterns" in data

    def test_load_missing_returns_empty(self, store, tmp_path):
        path = tmp_path / "does_not_exist.json"
        learner = PatternLearner(store=store, patterns_path=path)
        assert learner.pattern_count == 0
        assert learner.profile.total_code_corrections == 0

    def test_persistence_roundtrip(self, store, tmp_path):
        path = tmp_path / "patterns.json"
        learner1 = PatternLearner(store=store, patterns_path=path)
        for _ in range(3):
            learner1.learn_from_correction("def f(): pass", "def f(): pass\n")

        learner2 = PatternLearner(store=store, patterns_path=path)
        pat = next(
            p for p in learner2.profile.patterns if p.name == "trailing_newline"
        )
        assert pat.occurrences == 3

    def test_load_corrupt_json_returns_empty(self, store, tmp_path):
        path = tmp_path / "patterns.json"
        path.write_text("{ not valid json")
        learner = PatternLearner(store=store, patterns_path=path)
        assert learner.pattern_count == 0

    def test_load_missing_key_returns_empty(self, store, tmp_path):
        path = tmp_path / "patterns.json"
        # Pattern entry missing required "name" key -> KeyError -> empty profile.
        path.write_text(json.dumps({"patterns": [{"description": "no name"}]}))
        learner = PatternLearner(store=store, patterns_path=path)
        assert learner.pattern_count == 0

    def test_load_full_profile_from_disk(self, store, tmp_path):
        path = tmp_path / "patterns.json"
        payload = {
            "naming": {"snake_case": 0.8},
            "patterns": [
                {
                    "name": "trailing_newline",
                    "description": "Add trailing newline.",
                    "examples": ["x\n"],
                    "counter_examples": [],
                    "confidence": 0.9,
                    "occurrences": 9,
                }
            ],
            "language_preferences": {"python": True},
            "last_updated": 42.0,
            "total_code_corrections": 9,
        }
        path.write_text(json.dumps(payload))
        learner = PatternLearner(store=store, patterns_path=path)
        assert learner.profile.naming == {"snake_case": 0.8}
        assert learner.profile.language_preferences == {"python": True}
        assert learner.profile.last_updated == 42.0
        assert learner.profile.total_code_corrections == 9
        assert learner.pattern_count == 1
        assert learner.high_confidence_patterns[0].name == "trailing_newline"
