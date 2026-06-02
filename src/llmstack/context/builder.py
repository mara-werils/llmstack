"""Smart context builder — intelligently select the most relevant code for LLM prompts."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContextChunk:
    """A chunk of context with relevance metadata."""
    file: str
    content: str
    relevance: float  # 0-1
    reason: str
    line_start: int
    line_end: int
    tokens_estimate: int


class ContextBuilder:
    """Build optimized context for LLM prompts from a codebase."""

    IGNORE_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", "dist", "build", ".tox", ".eggs"}
    CODE_EXTS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".rb", ".php", ".swift", ".kt"}

    def __init__(self, directory: Path, max_tokens: int = 8000):
        self.directory = directory
        self.max_tokens = max_tokens

    def build(self, query: str, strategy: str = "smart") -> list[ContextChunk]:
        """Build context chunks optimized for the query."""
        strategies = {
            "smart": self._smart_context,
            "git": self._git_context,
            "imports": self._import_context,
            "related": self._related_context,
        }
        builder = strategies.get(strategy, self._smart_context)
        chunks = builder(query)

        # Fit within token budget
        return self._fit_budget(chunks)

    def _smart_context(self, query: str) -> list[ContextChunk]:
        """Smart context: combine keyword matching, git history, and structure."""
        chunks = []
        keywords = set(query.lower().split())

        for file_path in self._iter_code_files():
            try:
                content = file_path.read_text(errors="replace")
            except OSError:
                continue

            lines = content.split("\n")
            relevance = self._score_relevance(content, file_path, keywords)

            if relevance > 0.1:
                # Extract most relevant section
                best_start, best_end = self._find_best_section(lines, keywords)
                section = "\n".join(lines[best_start:best_end])

                chunks.append(ContextChunk(
                    file=str(file_path.relative_to(self.directory)),
                    content=section,
                    relevance=relevance,
                    reason="keyword match + structure analysis",
                    line_start=best_start + 1,
                    line_end=best_end,
                    tokens_estimate=len(section) // 4,
                ))

        return sorted(chunks, key=lambda c: -c.relevance)

    def _git_context(self, query: str) -> list[ContextChunk]:
        """Context from recently modified files."""
        chunks = []

        try:
            result = subprocess.run(
                ["git", "log", "--name-only", "--format=", "-20"],
                capture_output=True, text=True, cwd=str(self.directory), timeout=10,
            )
            recent_files = set(result.stdout.strip().split("\n")) if result.returncode == 0 else set()
        except Exception:
            recent_files = set()

        keywords = set(query.lower().split())

        for rel_path in recent_files:
            if not rel_path:
                continue
            file_path = self.directory / rel_path
            if not file_path.exists() or file_path.suffix not in self.CODE_EXTS:
                continue

            try:
                content = file_path.read_text(errors="replace")
            except OSError:
                continue

            relevance = self._score_relevance(content, file_path, keywords) + 0.2  # Boost for recency

            chunks.append(ContextChunk(
                file=rel_path,
                content=content[:3000],
                relevance=min(1.0, relevance),
                reason="recently modified",
                line_start=1,
                line_end=content.count("\n") + 1,
                tokens_estimate=len(content[:3000]) // 4,
            ))

        return sorted(chunks, key=lambda c: -c.relevance)

    def _import_context(self, query: str) -> list[ContextChunk]:
        """Context based on import relationships."""
        chunks = []
        keywords = set(query.lower().split())

        # Find the most relevant file
        best_file = None
        best_score = 0

        for file_path in self._iter_code_files():
            try:
                content = file_path.read_text(errors="replace")
            except OSError:
                continue
            score = self._score_relevance(content, file_path, keywords)
            if score > best_score:
                best_score = score
                best_file = file_path

        if not best_file:
            return chunks

        # Find imported modules
        try:
            tree = ast.parse(best_file.read_text(errors="replace"))
        except (SyntaxError, OSError):
            return chunks

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

        # Find files matching imported modules
        for file_path in self._iter_code_files():
            rel = str(file_path.relative_to(self.directory))
            module_name = file_path.stem
            if module_name in imported_modules or any(m in rel for m in imported_modules):
                try:
                    content = file_path.read_text(errors="replace")[:3000]
                except OSError:
                    continue

                chunks.append(ContextChunk(
                    file=rel,
                    content=content,
                    relevance=0.7,
                    reason=f"imported by {best_file.name}",
                    line_start=1,
                    line_end=content.count("\n") + 1,
                    tokens_estimate=len(content) // 4,
                ))

        return chunks

    def _related_context(self, query: str) -> list[ContextChunk]:
        """Context from structurally related files."""
        chunks = []
        keywords = set(query.lower().split())

        # Find files with matching names/patterns
        for file_path in self._iter_code_files():
            rel = str(file_path.relative_to(self.directory)).lower()
            name_matches = sum(1 for kw in keywords if kw in rel)

            if name_matches > 0:
                try:
                    content = file_path.read_text(errors="replace")[:3000]
                except OSError:
                    continue

                chunks.append(ContextChunk(
                    file=str(file_path.relative_to(self.directory)),
                    content=content,
                    relevance=0.5 + (name_matches * 0.15),
                    reason="filename match",
                    line_start=1,
                    line_end=content.count("\n") + 1,
                    tokens_estimate=len(content) // 4,
                ))

        return sorted(chunks, key=lambda c: -c.relevance)

    def _score_relevance(self, content: str, file_path: Path, keywords: set[str]) -> float:
        """Score content relevance to query keywords."""
        content_lower = content.lower()
        file_lower = str(file_path).lower()

        # Keyword matches in content
        content_matches = sum(1 for kw in keywords if kw in content_lower)
        # Keyword matches in filename
        name_matches = sum(1 for kw in keywords if kw in file_lower)

        score = 0.0
        if keywords:
            score += (content_matches / len(keywords)) * 0.6
            score += (name_matches / len(keywords)) * 0.4

        return min(1.0, score)

    def _find_best_section(self, lines: list[str], keywords: set[str], window: int = 50) -> tuple[int, int]:
        """Find the most relevant section of a file."""
        if len(lines) <= window:
            return 0, len(lines)

        best_score = -1
        best_start = 0

        for i in range(0, len(lines) - window, 10):
            section = "\n".join(lines[i:i + window]).lower()
            score = sum(1 for kw in keywords if kw in section)
            if score > best_score:
                best_score = score
                best_start = i

        return best_start, min(len(lines), best_start + window)

    def _fit_budget(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        """Fit chunks within token budget."""
        result = []
        total_tokens = 0

        for chunk in chunks:
            if total_tokens + chunk.tokens_estimate > self.max_tokens:
                # Try to truncate
                remaining = self.max_tokens - total_tokens
                if remaining > 100:
                    chars = remaining * 4
                    chunk.content = chunk.content[:chars]
                    chunk.tokens_estimate = remaining
                    result.append(chunk)
                break
            result.append(chunk)
            total_tokens += chunk.tokens_estimate

        return result

    def _iter_code_files(self):
        """Iterate over code files in directory."""
        for p in sorted(self.directory.rglob("*")):
            if p.is_file() and p.suffix in self.CODE_EXTS and not any(
                part in self.IGNORE_DIRS for part in p.parts
            ):
                yield p
