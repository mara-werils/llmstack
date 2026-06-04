"""MCP tool definitions for the learning pipeline.

Exposes learning pipeline functionality as MCP tools that can be
called by IDE clients (Claude Code, Cursor, VS Code, etc.).
"""

from __future__ import annotations

from typing import Any


LEARN_TOOLS = [
    {
        "name": "llmstack_feedback",
        "description": (
            "Submit feedback on an LLM response to improve future outputs. "
            "Supports thumbs_up, thumbs_down, correction, and edit types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "feedback_type": {
                    "type": "string",
                    "enum": ["thumbs_up", "thumbs_down", "correction", "edit"],
                    "description": "Type of feedback signal",
                },
                "query": {
                    "type": "string",
                    "description": "The original query/prompt",
                },
                "response": {
                    "type": "string",
                    "description": "The AI response being judged",
                },
                "correction": {
                    "type": "string",
                    "description": "The user's preferred/corrected response (for correction/edit types)",
                },
            },
            "required": ["feedback_type"],
        },
    },
    {
        "name": "llmstack_learn_status",
        "description": (
            "Get the current status of the adaptive learning pipeline, "
            "including feedback counts, model versions, and quality metrics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "llmstack_learn_preferences",
        "description": (
            "Get the user's learned preferences (response length, formatting, "
            "tone) that the AI has adapted to over time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def handle_learn_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a learning pipeline MCP tool call.

    Returns the tool result as a dict.
    """
    if name == "llmstack_feedback":
        return _handle_feedback(arguments)
    elif name == "llmstack_learn_status":
        return _handle_status()
    elif name == "llmstack_learn_preferences":
        return _handle_preferences()
    return {"error": f"Unknown learn tool: {name}"}


def _handle_feedback(args: dict[str, Any]) -> dict[str, Any]:
    """Handle feedback submission."""
    from llmstack.learn.feedback import Feedback, FeedbackType
    from llmstack.learn.store import FeedbackStore

    try:
        fb_type = FeedbackType(args.get("feedback_type", "thumbs_up"))
    except ValueError:
        return {"error": "Invalid feedback_type"}

    store = FeedbackStore()
    feedback = Feedback(
        feedback_type=fb_type,
        query=args.get("query", ""),
        response=args.get("response", ""),
        correction=args.get("correction", ""),
        command="mcp",
    )
    store.add_feedback(feedback)

    # Update preference learner if correction
    if feedback.has_correction:
        from llmstack.learn.preferences import PreferenceLearner

        learner = PreferenceLearner(store=store)
        learner.learn_from_feedback(feedback)

    pending = store.get_unused_feedback_count()
    store.close()

    return {
        "status": "recorded",
        "feedback_id": feedback.id,
        "pending_for_training": pending,
    }


def _handle_status() -> dict[str, Any]:
    """Handle status query."""
    from llmstack.learn.analytics import LearningAnalytics
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)
    analytics = LearningAnalytics(store=store, version_mgr=version_mgr)
    summary = analytics.get_summary()
    store.close()
    return summary


def _handle_preferences() -> dict[str, Any]:
    """Handle preferences query."""
    from llmstack.learn.preferences import PreferenceLearner
    from llmstack.learn.store import FeedbackStore

    store = FeedbackStore()
    learner = PreferenceLearner(store=store)
    profile = learner.get_profile()
    additions = learner.get_system_prompt_additions()
    store.close()

    return {
        "preferences": profile,
        "system_prompt_additions": additions,
    }
