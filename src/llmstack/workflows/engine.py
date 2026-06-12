"""Workflow engine — chain multiple llmstack commands into automated pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


BUILTIN_WORKFLOWS = {
    "pr-review": {
        "name": "PR Review Pipeline",
        "description": "Full code quality check before PR",
        "steps": [
            {"name": "complexity", "command": "complexity", "args": {"target": "."}},
            {"name": "dead-code", "command": "dead-code", "args": {"target": "."}},
            {"name": "security", "command": "security", "args": {"target": "."}},
            {"name": "review", "command": "review", "args": {"staged": True}},
        ],
    },
    "code-health": {
        "name": "Code Health Check",
        "description": "Comprehensive code quality analysis",
        "steps": [
            {
                "name": "complexity",
                "command": "complexity",
                "args": {"target": ".", "show_all": True},
            },
            {"name": "dead-code", "command": "dead-code", "args": {"target": "."}},
            {"name": "tokens", "command": "tokens", "args": {"target": "."}},
            {"name": "deps", "command": "deps", "args": {}},
        ],
    },
    "onboard": {
        "name": "Project Onboarding",
        "description": "Understand a new codebase quickly",
        "steps": [
            {"name": "info", "command": "info", "args": {}},
            {"name": "tokens", "command": "tokens", "args": {"target": "."}},
            {"name": "complexity", "command": "complexity", "args": {"target": "."}},
            {
                "name": "diagram",
                "command": "diagram",
                "args": {"target": ".", "diagram_type": "architecture"},
            },
        ],
    },
    "ship-it": {
        "name": "Ship It Pipeline",
        "description": "Review, test, commit, and prepare for push",
        "steps": [
            {"name": "review", "command": "review", "args": {"staged": True}},
            {"name": "security", "command": "security", "args": {"target": "."}},
            {"name": "commit", "command": "commit", "args": {}},
        ],
    },
    "daily-digest": {
        "name": "Daily Digest",
        "description": "Summarize recent changes and code health",
        "steps": [
            {"name": "changelog", "command": "changelog", "args": {"max_commits": 20}},
            {"name": "complexity", "command": "complexity", "args": {"target": "."}},
        ],
    },
}


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    name: str
    command: str
    args: dict = field(default_factory=dict)
    continue_on_error: bool = True


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    name: str
    total_steps: int
    completed: int
    failed: int
    skipped: int
    duration: float
    step_results: list[dict]


class WorkflowEngine:
    """Executes workflow pipelines."""

    def __init__(self):
        self.custom_workflows: dict[str, dict] = {}
        self._load_custom()

    @property
    def workflow_count(self) -> int:
        """Return the total number of available workflows."""
        return len(BUILTIN_WORKFLOWS) + len(self.custom_workflows)

    def has_workflow(self, name: str) -> bool:
        """Return True if a workflow with this name exists."""
        return name in BUILTIN_WORKFLOWS or name in self.custom_workflows

    def _load_custom(self) -> None:
        """Load custom workflows from ~/.llmstack/workflows/."""
        workflows_dir = Path.home() / ".llmstack" / "workflows"
        if not workflows_dir.exists():
            return

        for f in workflows_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self.custom_workflows[f.stem] = data
            except (json.JSONDecodeError, OSError):
                pass

    def list_workflows(self) -> list[dict]:
        """List all available workflows."""
        workflows = []
        for name, wf in BUILTIN_WORKFLOWS.items():
            workflows.append(
                {
                    "name": name,
                    "title": wf["name"],
                    "description": wf["description"],
                    "steps": len(wf["steps"]),
                    "builtin": True,
                }
            )
        for name, wf in self.custom_workflows.items():
            workflows.append(
                {
                    "name": name,
                    "title": wf.get("name", name),
                    "description": wf.get("description", ""),
                    "steps": len(wf.get("steps", [])),
                    "builtin": False,
                }
            )
        return workflows

    def get_workflow(self, name: str) -> dict | None:
        """Get workflow definition by name."""
        if name in BUILTIN_WORKFLOWS:
            return BUILTIN_WORKFLOWS[name]
        return self.custom_workflows.get(name)

    def save_custom(self, name: str, workflow: dict) -> None:
        """Save a custom workflow."""
        workflows_dir = Path.home() / ".llmstack" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        (workflows_dir / f"{name}.json").write_text(json.dumps(workflow, indent=2))
        self.custom_workflows[name] = workflow

    def delete_custom(self, name: str) -> bool:
        """Delete a custom workflow."""
        path = Path.home() / ".llmstack" / "workflows" / f"{name}.json"
        if path.exists():
            path.unlink()
            self.custom_workflows.pop(name, None)
            return True
        return False
