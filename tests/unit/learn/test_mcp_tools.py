"""Tests for the learning pipeline MCP tool definitions and dispatch."""

from __future__ import annotations

import pytest

from llmstack.learn import mcp_tools
from llmstack.learn.feedback import FeedbackType


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Redirect the default FeedbackStore DB path into tmp_path.

    The MCP tool handlers construct ``FeedbackStore()`` with no arguments,
    which defaults to ``~/.llmstack/learning.db``. Patching the module-level
    default keeps all I/O inside the test sandbox.
    """
    from llmstack.learn import store as store_mod

    db_path = tmp_path / "learning.db"
    monkeypatch.setattr(store_mod, "DEFAULT_DB_PATH", db_path)
    return db_path


class TestLearnToolsSchema:
    def test_three_tools_defined(self):
        assert len(mcp_tools.LEARN_TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in mcp_tools.LEARN_TOOLS}
        assert names == {
            "llmstack_feedback",
            "llmstack_learn_status",
            "llmstack_learn_preferences",
        }

    def test_every_tool_has_required_fields(self):
        for tool in mcp_tools.LEARN_TOOLS:
            assert isinstance(tool["name"], str) and tool["name"]
            assert isinstance(tool["description"], str) and tool["description"]
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_feedback_schema_enum_and_required(self):
        feedback_tool = next(
            t for t in mcp_tools.LEARN_TOOLS if t["name"] == "llmstack_feedback"
        )
        props = feedback_tool["inputSchema"]["properties"]
        assert set(props) == {"feedback_type", "query", "response", "correction"}
        assert props["feedback_type"]["enum"] == [
            "thumbs_up",
            "thumbs_down",
            "correction",
            "edit",
        ]
        assert feedback_tool["inputSchema"]["required"] == ["feedback_type"]

    def test_status_and_preferences_have_no_required_inputs(self):
        for name in ("llmstack_learn_status", "llmstack_learn_preferences"):
            tool = next(t for t in mcp_tools.LEARN_TOOLS if t["name"] == name)
            assert tool["inputSchema"]["properties"] == {}
            assert "required" not in tool["inputSchema"]


class TestHandleLearnToolDispatch:
    def test_unknown_tool_returns_error(self):
        result = mcp_tools.handle_learn_tool("does_not_exist", {})
        assert result == {"error": "Unknown learn tool: does_not_exist"}

    def test_dispatch_feedback(self, monkeypatch):
        sentinel = {"status": "recorded"}
        seen = {}

        def fake(args):
            seen["args"] = args
            return sentinel

        monkeypatch.setattr(mcp_tools, "_handle_feedback", fake)
        out = mcp_tools.handle_learn_tool("llmstack_feedback", {"feedback_type": "thumbs_up"})
        assert out is sentinel
        assert seen["args"] == {"feedback_type": "thumbs_up"}

    def test_dispatch_status(self, monkeypatch):
        sentinel = {"status": "ok"}
        monkeypatch.setattr(mcp_tools, "_handle_status", lambda: sentinel)
        assert mcp_tools.handle_learn_tool("llmstack_learn_status", {}) is sentinel

    def test_dispatch_preferences(self, monkeypatch):
        sentinel = {"preferences": {}}
        monkeypatch.setattr(mcp_tools, "_handle_preferences", lambda: sentinel)
        assert mcp_tools.handle_learn_tool("llmstack_learn_preferences", {}) is sentinel


class TestHandleFeedback:
    def test_invalid_feedback_type(self, tmp_store):
        result = mcp_tools.handle_learn_tool(
            "llmstack_feedback", {"feedback_type": "bogus"}
        )
        assert result == {"error": "Invalid feedback_type"}

    def test_thumbs_up_recorded(self, tmp_store):
        result = mcp_tools.handle_learn_tool(
            "llmstack_feedback",
            {
                "feedback_type": "thumbs_up",
                "query": "What is Python?",
                "response": "A language.",
            },
        )
        assert result["status"] == "recorded"
        assert isinstance(result["feedback_id"], str) and result["feedback_id"]
        assert result["pending_for_training"] == 1

    def test_default_feedback_type_when_missing(self, tmp_store):
        # Missing feedback_type defaults to thumbs_up inside the handler.
        result = mcp_tools.handle_learn_tool("llmstack_feedback", {})
        assert result["status"] == "recorded"
        assert result["pending_for_training"] == 1

    def test_pending_count_increments(self, tmp_store):
        for _ in range(3):
            mcp_tools.handle_learn_tool(
                "llmstack_feedback",
                {"feedback_type": "thumbs_up", "query": "q", "response": "r"},
            )
        final = mcp_tools.handle_learn_tool(
            "llmstack_feedback",
            {"feedback_type": "thumbs_up", "query": "q", "response": "r"},
        )
        assert final["pending_for_training"] == 4

    def test_correction_triggers_preference_learning(self, tmp_store, monkeypatch):
        # Confirm the correction branch (has_correction) runs the learner.
        learned = {}

        from llmstack.learn import preferences as pref_mod

        real_learn = pref_mod.PreferenceLearner.learn_from_feedback

        def spy(self, feedback):
            learned["called"] = True
            learned["type"] = feedback.feedback_type
            return real_learn(self, feedback)

        monkeypatch.setattr(pref_mod.PreferenceLearner, "learn_from_feedback", spy)

        result = mcp_tools.handle_learn_tool(
            "llmstack_feedback",
            {
                "feedback_type": "correction",
                "query": "Write hello world",
                "response": "print(1)",
                "correction": "print('Hello, World!')",
            },
        )
        assert result["status"] == "recorded"
        assert learned.get("called") is True
        assert learned["type"] == FeedbackType.CORRECTION

    def test_correction_without_text_skips_learning(self, tmp_store, monkeypatch):
        # feedback_type=correction but empty correction => has_correction is False,
        # so the preference learner must NOT be invoked.
        from llmstack.learn import preferences as pref_mod

        def boom(self, feedback):  # pragma: no cover - must not run
            raise AssertionError("learner should not be called")

        monkeypatch.setattr(pref_mod.PreferenceLearner, "learn_from_feedback", boom)

        result = mcp_tools.handle_learn_tool(
            "llmstack_feedback",
            {"feedback_type": "correction", "query": "q", "response": "r"},
        )
        assert result["status"] == "recorded"


class TestHandleStatus:
    def test_status_returns_summary_dict(self, tmp_store):
        result = mcp_tools.handle_learn_tool("llmstack_learn_status", {})
        assert isinstance(result, dict)
        # get_summary always reports a status field on an empty pipeline.
        assert "status" in result

    def test_status_after_feedback(self, tmp_store):
        mcp_tools.handle_learn_tool(
            "llmstack_feedback",
            {"feedback_type": "thumbs_up", "query": "q", "response": "r"},
        )
        result = mcp_tools.handle_learn_tool("llmstack_learn_status", {})
        assert isinstance(result, dict)


class TestHandlePreferences:
    def test_preferences_shape(self, tmp_store):
        result = mcp_tools.handle_learn_tool("llmstack_learn_preferences", {})
        assert set(result) == {"preferences", "system_prompt_additions"}
        assert isinstance(result["preferences"], dict)
        assert isinstance(result["system_prompt_additions"], str)

    def test_preferences_direct_helper(self, tmp_store):
        # Exercise the private helper directly as well.
        result = mcp_tools._handle_preferences()
        assert "preferences" in result
        assert "system_prompt_additions" in result
