"""Prompt template management — store, version, and reuse prompt templates.

Templates support Jinja2-style variable substitution (``{{variable}}``)
and maintain a version history so you can roll back to any previous version.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class TemplateVersion:
    """A single version snapshot of a prompt template."""

    version: int
    content: str
    variables: list[str]
    created_at: float = 0.0
    author: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class PromptTemplate:
    """A named prompt template with version history."""

    id: str = ""
    name: str = ""
    description: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    current_version: int = 1
    versions: list[TemplateVersion] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def variable_count(self) -> int:
        """Return the number of variables in the current template version."""
        current = self.get_current()
        return len(current.variables) if current else 0

    @property
    def version_count(self) -> int:
        """Return how many versions this template has."""
        return len(self.versions)

    def get_current(self) -> TemplateVersion | None:
        """Return the current version of the template."""
        for v in self.versions:
            if v.version == self.current_version:
                return v
        return self.versions[-1] if self.versions else None

    def render(self, variables: dict[str, str] | None = None) -> str:
        """Render the current template with variable substitution."""
        current = self.get_current()
        if current is None:
            return ""
        content = current.content
        if variables:
            for key, value in variables.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
        return content

    def to_dict(self) -> dict[str, Any]:
        current = self.get_current()
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "current_version": self.current_version,
            "total_versions": len(self.versions),
            "content": current.content if current else "",
            "variables": current.variables if current else [],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _extract_variables(content: str) -> list[str]:
    """Extract ``{{variable}}`` placeholders from template content."""
    return sorted(set(re.findall(r"\{\{(\w+)\}\}", content)))


class TemplateStore:
    """In-memory store for prompt templates with CRUD and versioning."""

    def __init__(self):
        self._lock = Lock()
        self._templates: dict[str, PromptTemplate] = {}
        self._name_index: dict[str, str] = {}  # name -> id

    def create(
        self,
        name: str,
        content: str,
        description: str = "",
        category: str = "general",
        tags: list[str] | None = None,
        author: str = "",
    ) -> PromptTemplate:
        """Create a new prompt template."""
        with self._lock:
            if name in self._name_index:
                raise ValueError(f"Template '{name}' already exists")

            variables = _extract_variables(content)
            version = TemplateVersion(
                version=1,
                content=content,
                variables=variables,
                author=author,
            )

            template = PromptTemplate(
                name=name,
                description=description,
                category=category,
                tags=tags or [],
                current_version=1,
                versions=[version],
            )

            self._templates[template.id] = template
            self._name_index[name] = template.id
            return template

    def get(self, name_or_id: str) -> PromptTemplate | None:
        """Get a template by name or ID."""
        with self._lock:
            if name_or_id in self._templates:
                return self._templates[name_or_id]
            tid = self._name_index.get(name_or_id)
            if tid:
                return self._templates.get(tid)
            return None

    def update(
        self,
        name_or_id: str,
        content: str,
        author: str = "",
    ) -> PromptTemplate | None:
        """Update a template, creating a new version."""
        with self._lock:
            template = self._get_locked(name_or_id)
            if template is None:
                return None

            new_version_num = template.current_version + 1
            variables = _extract_variables(content)
            version = TemplateVersion(
                version=new_version_num,
                content=content,
                variables=variables,
                author=author,
            )

            template.versions.append(version)
            template.current_version = new_version_num
            template.updated_at = time.time()
            return template

    def delete(self, name_or_id: str) -> bool:
        """Delete a template by name or ID."""
        with self._lock:
            template = self._get_locked(name_or_id)
            if template is None:
                return False

            del self._templates[template.id]
            self._name_index.pop(template.name, None)
            return True

    def list_all(
        self,
        category: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> list[PromptTemplate]:
        """List templates with optional filters."""
        with self._lock:
            results = []
            for t in self._templates.values():
                if category and t.category != category:
                    continue
                if tag and tag not in t.tags:
                    continue
                results.append(t)
                if len(results) >= limit:
                    break
            return sorted(results, key=lambda t: t.updated_at, reverse=True)

    def rollback(self, name_or_id: str, version: int) -> PromptTemplate | None:
        """Roll back a template to a specific version."""
        with self._lock:
            template = self._get_locked(name_or_id)
            if template is None:
                return None

            valid_versions = {v.version for v in template.versions}
            if version not in valid_versions:
                raise ValueError(f"Version {version} not found")

            template.current_version = version
            template.updated_at = time.time()
            return template

    def render(self, name_or_id: str, variables: dict[str, str] | None = None) -> str | None:
        """Render a template by name/ID with variable substitution."""
        template = self.get(name_or_id)
        if template is None:
            return None
        return template.render(variables)

    def search(self, query: str, limit: int = 20) -> list[PromptTemplate]:
        """Simple text search across template names and descriptions."""
        with self._lock:
            query_lower = query.lower()
            results = []
            for t in self._templates.values():
                if (
                    query_lower in t.name.lower()
                    or query_lower in t.description.lower()
                    or any(query_lower in tag.lower() for tag in t.tags)
                ):
                    results.append(t)
                    if len(results) >= limit:
                        break
            return results

    def _get_locked(self, name_or_id: str) -> PromptTemplate | None:
        """Internal lookup (caller must hold _lock)."""
        if name_or_id in self._templates:
            return self._templates[name_or_id]
        tid = self._name_index.get(name_or_id)
        if tid:
            return self._templates.get(tid)
        return None

    @property
    def count(self) -> int:
        return len(self._templates)


# Built-in templates that ship with LLMStack
BUILTIN_TEMPLATES = [
    {
        "name": "code-review",
        "description": "Code review prompt for analyzing diffs",
        "category": "development",
        "tags": ["code", "review", "development"],
        "content": (
            "You are a senior software engineer performing a code review.\n\n"
            "## Code to Review\n```\n{{code}}\n```\n\n"
            "## Language: {{language}}\n\n"
            "Provide feedback on:\n"
            "1. Code quality and readability\n"
            "2. Potential bugs or edge cases\n"
            "3. Performance considerations\n"
            "4. Security concerns\n"
            "5. Suggested improvements"
        ),
    },
    {
        "name": "summarize",
        "description": "Summarize text to a target length",
        "category": "general",
        "tags": ["summary", "text"],
        "content": (
            "Summarize the following text in {{length}} sentences.\n"
            "Focus on the key points and maintain accuracy.\n\n"
            "Text:\n{{text}}"
        ),
    },
    {
        "name": "explain-code",
        "description": "Explain code in plain language",
        "category": "development",
        "tags": ["code", "explanation", "education"],
        "content": (
            "Explain the following {{language}} code in plain English.\n"
            "Target audience: {{audience}}\n\n"
            "```{{language}}\n{{code}}\n```\n\n"
            "Explain:\n"
            "1. What the code does (high-level overview)\n"
            "2. How it works (step by step)\n"
            "3. Any notable patterns or techniques used"
        ),
    },
    {
        "name": "sql-query",
        "description": "Generate SQL query from natural language",
        "category": "data",
        "tags": ["sql", "database", "query"],
        "content": (
            "Given the following database schema:\n\n"
            "{{schema}}\n\n"
            "Write a SQL query that: {{description}}\n\n"
            "Requirements:\n"
            "- Use standard SQL syntax\n"
            "- Optimize for readability\n"
            "- Include comments explaining complex joins or subqueries"
        ),
    },
    {
        "name": "test-generation",
        "description": "Generate unit tests for code",
        "category": "development",
        "tags": ["testing", "code", "quality"],
        "content": (
            "Generate comprehensive unit tests for the following {{language}} code "
            "using the {{framework}} testing framework.\n\n"
            "```{{language}}\n{{code}}\n```\n\n"
            "Include:\n"
            "- Happy path tests\n"
            "- Edge cases\n"
            "- Error handling tests\n"
            "- Boundary value tests"
        ),
    },
]
