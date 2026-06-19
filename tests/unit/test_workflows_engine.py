"""Tests for the workflow engine — llmstack.workflows.engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmstack import workflows
from llmstack.workflows.engine import (
    BUILTIN_WORKFLOWS,
    WorkflowEngine,
    WorkflowResult,
    WorkflowStep,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Redirect Path.home() to an isolated temp dir so no real FS is touched."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture
def engine(home):
    """A fresh engine rooted at the temp home (no custom workflows on disk)."""
    return WorkflowEngine()


class TestPackageExports:
    def test_workflow_engine_exported(self):
        assert workflows.WorkflowEngine is WorkflowEngine

    def test_all_contains_workflow_engine(self):
        assert workflows.__all__ == ["WorkflowEngine"]


class TestDataclasses:
    def test_step_defaults(self):
        step = WorkflowStep(name="s", command="complexity")
        assert step.name == "s"
        assert step.command == "complexity"
        assert step.args == {}
        assert step.continue_on_error is True

    def test_step_independent_args(self):
        a = WorkflowStep(name="a", command="x")
        b = WorkflowStep(name="b", command="y")
        a.args["k"] = 1
        assert b.args == {}

    def test_step_explicit_values(self):
        step = WorkflowStep(name="s", command="c", args={"target": "."}, continue_on_error=False)
        assert step.args == {"target": "."}
        assert step.continue_on_error is False

    def test_workflow_result_fields(self):
        result = WorkflowResult(
            name="pr-review",
            total_steps=4,
            completed=3,
            failed=1,
            skipped=0,
            duration=1.5,
            step_results=[{"name": "complexity"}],
        )
        assert result.name == "pr-review"
        assert result.total_steps == 4
        assert result.completed == 3
        assert result.failed == 1
        assert result.skipped == 0
        assert result.duration == 1.5
        assert result.step_results == [{"name": "complexity"}]


class TestBuiltins:
    def test_builtin_names_present(self):
        for name in ("pr-review", "code-health", "onboard", "ship-it", "daily-digest"):
            assert name in BUILTIN_WORKFLOWS

    def test_builtin_structure(self):
        for wf in BUILTIN_WORKFLOWS.values():
            assert "name" in wf
            assert "description" in wf
            assert isinstance(wf["steps"], list)
            assert wf["steps"]


class TestInitAndCounts:
    def test_init_no_custom_dir(self, engine):
        assert engine.custom_workflows == {}

    def test_workflow_count_builtin_only(self, engine):
        assert engine.workflow_count == len(BUILTIN_WORKFLOWS)

    def test_workflow_count_with_custom(self, engine):
        engine.custom_workflows["mine"] = {"name": "Mine", "steps": []}
        assert engine.workflow_count == len(BUILTIN_WORKFLOWS) + 1


class TestHasWorkflow:
    def test_has_builtin(self, engine):
        assert engine.has_workflow("pr-review") is True

    def test_has_custom(self, engine):
        engine.custom_workflows["mine"] = {"name": "Mine", "steps": []}
        assert engine.has_workflow("mine") is True

    def test_missing(self, engine):
        assert engine.has_workflow("nope") is False


class TestLoadCustom:
    def test_no_dir_loads_nothing(self, engine):
        # home tmp_path has no ~/.llmstack/workflows directory
        assert engine.custom_workflows == {}

    def test_loads_valid_json(self, home):
        wdir = home / ".llmstack" / "workflows"
        wdir.mkdir(parents=True)
        payload = {"name": "Custom", "description": "d", "steps": [{"name": "s"}]}
        (wdir / "custom.json").write_text(json.dumps(payload))

        engine = WorkflowEngine()
        assert "custom" in engine.custom_workflows
        assert engine.custom_workflows["custom"] == payload

    def test_skips_invalid_json(self, home):
        wdir = home / ".llmstack" / "workflows"
        wdir.mkdir(parents=True)
        (wdir / "bad.json").write_text("{not valid json")
        (wdir / "good.json").write_text(json.dumps({"name": "ok", "steps": []}))

        engine = WorkflowEngine()
        assert "bad" not in engine.custom_workflows
        assert "good" in engine.custom_workflows

    def test_skips_unreadable_file(self, home, monkeypatch):
        wdir = home / ".llmstack" / "workflows"
        wdir.mkdir(parents=True)
        (wdir / "x.json").write_text("{}")

        orig_read_text = Path.read_text

        def boom(self, *args, **kwargs):
            if self.name == "x.json":
                raise OSError("cannot read")
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", boom)
        engine = WorkflowEngine()
        assert "x" not in engine.custom_workflows


class TestListWorkflows:
    def test_lists_all_builtins(self, engine):
        listed = engine.list_workflows()
        assert len(listed) == len(BUILTIN_WORKFLOWS)
        names = {w["name"] for w in listed}
        assert names == set(BUILTIN_WORKFLOWS)

    def test_builtin_entry_shape(self, engine):
        entry = next(w for w in engine.list_workflows() if w["name"] == "pr-review")
        assert entry["title"] == "PR Review Pipeline"
        assert entry["description"] == "Full code quality check before PR"
        assert entry["steps"] == 4
        assert entry["builtin"] is True

    def test_includes_custom(self, engine):
        engine.custom_workflows["mine"] = {
            "name": "My Flow",
            "description": "desc",
            "steps": [{"name": "a"}, {"name": "b"}],
        }
        entry = next(w for w in engine.list_workflows() if w["name"] == "mine")
        assert entry["title"] == "My Flow"
        assert entry["description"] == "desc"
        assert entry["steps"] == 2
        assert entry["builtin"] is False

    def test_custom_defaults_when_keys_missing(self, engine):
        engine.custom_workflows["bare"] = {}
        entry = next(w for w in engine.list_workflows() if w["name"] == "bare")
        # title falls back to the workflow name, description to "", steps to 0
        assert entry["title"] == "bare"
        assert entry["description"] == ""
        assert entry["steps"] == 0
        assert entry["builtin"] is False


class TestGetWorkflow:
    def test_get_builtin(self, engine):
        wf = engine.get_workflow("pr-review")
        assert wf is BUILTIN_WORKFLOWS["pr-review"]

    def test_get_custom(self, engine):
        custom = {"name": "Mine", "steps": []}
        engine.custom_workflows["mine"] = custom
        assert engine.get_workflow("mine") is custom

    def test_get_missing_returns_none(self, engine):
        assert engine.get_workflow("does-not-exist") is None


class TestSaveCustom:
    def test_save_writes_file_and_registers(self, home, engine):
        wf = {"name": "Saved", "description": "d", "steps": []}
        engine.save_custom("saved", wf)

        path = home / ".llmstack" / "workflows" / "saved.json"
        assert path.exists()
        assert json.loads(path.read_text()) == wf
        assert engine.custom_workflows["saved"] == wf
        assert engine.has_workflow("saved") is True

    def test_save_creates_missing_dirs(self, home, engine):
        # workflows dir does not exist yet
        assert not (home / ".llmstack" / "workflows").exists()
        engine.save_custom("a", {"name": "A", "steps": []})
        assert (home / ".llmstack" / "workflows" / "a.json").exists()

    def test_save_overwrites(self, home, engine):
        engine.save_custom("a", {"name": "v1", "steps": []})
        engine.save_custom("a", {"name": "v2", "steps": []})
        path = home / ".llmstack" / "workflows" / "a.json"
        assert json.loads(path.read_text())["name"] == "v2"
        assert engine.custom_workflows["a"]["name"] == "v2"


class TestDeleteCustom:
    def test_delete_existing(self, home, engine):
        engine.save_custom("a", {"name": "A", "steps": []})
        path = home / ".llmstack" / "workflows" / "a.json"
        assert path.exists()

        assert engine.delete_custom("a") is True
        assert not path.exists()
        assert "a" not in engine.custom_workflows

    def test_delete_missing_returns_false(self, engine):
        assert engine.delete_custom("ghost") is False

    def test_delete_file_present_but_not_in_dict(self, home, engine):
        # File exists on disk but not registered in custom_workflows.
        wdir = home / ".llmstack" / "workflows"
        wdir.mkdir(parents=True)
        (wdir / "orphan.json").write_text("{}")
        assert "orphan" not in engine.custom_workflows
        assert engine.delete_custom("orphan") is True
        assert not (wdir / "orphan.json").exists()


class TestRoundTrip:
    def test_save_then_reload_in_new_engine(self, home, engine):
        wf = {"name": "Persisted", "description": "x", "steps": [{"name": "s"}]}
        engine.save_custom("persisted", wf)

        fresh = WorkflowEngine()
        assert fresh.has_workflow("persisted") is True
        assert fresh.get_workflow("persisted") == wf
