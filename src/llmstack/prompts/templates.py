"""Prompt template engine — reusable, parameterized prompts for common tasks."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Built-in templates for common development tasks
BUILTIN_TEMPLATES: list[dict] = [
    {
        "name": "code-review",
        "description": "Thorough code review with severity levels",
        "category": "development",
        "template": "Review this code for bugs, security issues, and improvements:\n\n```{{language}}\n{{code}}\n```\n\nFocus on: {{focus|security, performance, readability}}",
        "variables": ["language", "code", "focus"],
    },
    {
        "name": "unit-test",
        "description": "Generate unit tests for a function or class",
        "category": "testing",
        "template": "Generate comprehensive unit tests for this {{language}} code using {{framework|pytest}}:\n\n```{{language}}\n{{code}}\n```\n\nInclude edge cases, error cases, and happy path tests.",
        "variables": ["language", "code", "framework"],
    },
    {
        "name": "docstring",
        "description": "Generate docstrings for functions/classes",
        "category": "documentation",
        "template": "Generate {{style|Google}} style docstrings for all functions and classes in this code:\n\n```{{language}}\n{{code}}\n```\n\nInclude parameter descriptions, return types, and examples.",
        "variables": ["language", "code", "style"],
    },
    {
        "name": "debug",
        "description": "Debug an error or issue",
        "category": "debugging",
        "template": "I'm getting this error:\n\n```\n{{error}}\n```\n\nIn this code:\n\n```{{language}}\n{{code}}\n```\n\nExplain the root cause and provide a fix.",
        "variables": ["error", "language", "code"],
    },
    {
        "name": "optimize",
        "description": "Optimize code for performance",
        "category": "performance",
        "template": "Optimize this {{language}} code for {{goal|speed}}:\n\n```{{language}}\n{{code}}\n```\n\nExplain each optimization and its expected impact.",
        "variables": ["language", "code", "goal"],
    },
    {
        "name": "api-design",
        "description": "Design a REST API endpoint",
        "category": "architecture",
        "template": "Design a REST API for {{feature}}.\n\nRequirements:\n{{requirements}}\n\nInclude: endpoint paths, HTTP methods, request/response schemas, error codes, and authentication.",
        "variables": ["feature", "requirements"],
    },
    {
        "name": "sql-query",
        "description": "Generate SQL queries from natural language",
        "category": "database",
        "template": "Given this database schema:\n\n```sql\n{{schema}}\n```\n\nWrite a SQL query to: {{query_description}}\n\nOptimize for {{db_type|PostgreSQL}} and include indexes if needed.",
        "variables": ["schema", "query_description", "db_type"],
    },
    {
        "name": "regex",
        "description": "Generate and explain regex patterns",
        "category": "utility",
        "template": "Create a regex pattern in {{language|python}} that matches: {{description}}\n\nExamples of what should match:\n{{examples}}\n\nProvide the regex, explanation of each part, and test cases.",
        "variables": ["language", "description", "examples"],
    },
    {
        "name": "commit-msg",
        "description": "Generate conventional commit messages",
        "category": "git",
        "template": "Generate a conventional commit message for this diff:\n\n```diff\n{{diff}}\n```\n\nUse format: type(scope): description\nTypes: feat, fix, docs, style, refactor, test, chore",
        "variables": ["diff"],
    },
    {
        "name": "migration",
        "description": "Generate database migration scripts",
        "category": "database",
        "template": "Generate a database migration script for {{db_type|PostgreSQL}}.\n\nCurrent schema:\n```sql\n{{current_schema}}\n```\n\nDesired changes:\n{{changes}}\n\nInclude both UP and DOWN migrations.",
        "variables": ["db_type", "current_schema", "changes"],
    },
    {
        "name": "convert-types",
        "description": "Add type annotations to untyped code",
        "category": "development",
        "template": "Add complete type annotations to this {{language}} code:\n\n```{{language}}\n{{code}}\n```\n\nUse modern type syntax. Add TypedDict, generics, and Union types where appropriate.",
        "variables": ["language", "code"],
    },
    {
        "name": "readme",
        "description": "Generate a project README",
        "category": "documentation",
        "template": "Generate a comprehensive README.md for a project called {{project_name}}.\n\nDescription: {{description}}\nTech stack: {{tech_stack}}\nKey features: {{features}}\n\nInclude: badges, installation, usage, API reference, contributing guide.",
        "variables": ["project_name", "description", "tech_stack", "features"],
    },
]


@dataclass
class PromptTemplate:
    """A prompt template with variables."""
    name: str
    description: str
    category: str
    template: str
    variables: list[str]
    is_builtin: bool = False
    created_at: float = 0.0
    usage_count: int = 0


class TemplateManager:
    """Manages prompt templates with SQLite storage."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            data_dir = Path.home() / ".llmstack" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "templates.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    name TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    category TEXT DEFAULT '',
                    template TEXT NOT NULL,
                    variables TEXT DEFAULT '[]',
                    is_builtin INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def render(self, name: str, **variables) -> str:
        """Render a template with variables."""
        template = self.get(name)
        if not template:
            raise ValueError(f"Template not found: {name}")

        # Increment usage
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE templates SET usage_count = usage_count + 1 WHERE name = ?",
                (name,),
            )
            conn.commit()

        result = template.template
        # Replace {{var|default}} patterns
        for match in re.finditer(r"\{\{(\w+)(?:\|([^}]*))?\}\}", result):
            var_name = match.group(1)
            default_val = match.group(2) or ""
            value = variables.get(var_name, default_val)
            result = result.replace(match.group(0), str(value))

        return result

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name (checks DB first, then builtins)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM templates WHERE name = ?", (name,)).fetchone()
            if row:
                return self._row_to_template(row)

        # Check builtins
        for bt in BUILTIN_TEMPLATES:
            if bt["name"] == name:
                return PromptTemplate(
                    name=bt["name"],
                    description=bt["description"],
                    category=bt["category"],
                    template=bt["template"],
                    variables=bt["variables"],
                    is_builtin=True,
                )
        return None

    def save(self, name: str, template: str, description: str = "",
             category: str = "custom", variables: list[str] | None = None) -> PromptTemplate:
        """Save a custom template."""
        # Auto-detect variables from {{var}} patterns
        if variables is None:
            variables = list(set(re.findall(r"\{\{(\w+)(?:\|[^}]*)?\}\}", template)))

        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO templates
                   (name, description, category, template, variables, is_builtin, created_at, usage_count)
                   VALUES (?, ?, ?, ?, ?, 0, ?, 0)""",
                (name, description, category, template, json.dumps(variables), now),
            )
            conn.commit()

        return PromptTemplate(
            name=name, description=description, category=category,
            template=template, variables=variables, created_at=now,
        )

    def list_all(self, category: str | None = None) -> list[PromptTemplate]:
        """List all templates (builtin + custom)."""
        templates = []

        # Add builtins
        for bt in BUILTIN_TEMPLATES:
            if category and bt["category"] != category:
                continue
            templates.append(PromptTemplate(
                name=bt["name"], description=bt["description"],
                category=bt["category"], template=bt["template"],
                variables=bt["variables"], is_builtin=True,
            ))

        # Add custom from DB
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM templates WHERE is_builtin = 0"
            params = []
            if category:
                query += " AND category = ?"
                params.append(category)
            query += " ORDER BY usage_count DESC"

            for row in conn.execute(query, params).fetchall():
                templates.append(self._row_to_template(row))

        return templates

    def delete(self, name: str) -> bool:
        """Delete a custom template."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM templates WHERE name = ? AND is_builtin = 0", (name,))
            conn.commit()
            return conn.total_changes > 0

    def categories(self) -> list[str]:
        """List all categories."""
        cats = set(bt["category"] for bt in BUILTIN_TEMPLATES)
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT DISTINCT category FROM templates").fetchall():
                cats.add(row[0])
        return sorted(cats)

    def _row_to_template(self, row) -> PromptTemplate:
        return PromptTemplate(
            name=row["name"],
            description=row["description"],
            category=row["category"],
            template=row["template"],
            variables=json.loads(row["variables"]),
            is_builtin=bool(row["is_builtin"]),
            created_at=row["created_at"],
            usage_count=row["usage_count"],
        )
