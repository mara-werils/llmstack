"""Snippet manager — save, search, and reuse code snippets locally."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Snippet:
    """A saved code snippet."""

    id: str
    title: str
    code: str
    language: str
    tags: list[str]
    description: str
    source_file: str
    created_at: float
    updated_at: float
    usage_count: int = 0


class SnippetManager:
    """SQLite-backed snippet storage with full-text search."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            data_dir = Path.home() / ".llmstack" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "snippets.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snippets (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    code TEXT NOT NULL,
                    language TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    description TEXT DEFAULT '',
                    source_file TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snippet_fts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL
                )
            """)
            conn.commit()

    def _generate_id(self) -> str:
        import hashlib

        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]

    def save(
        self,
        title: str,
        code: str,
        language: str = "",
        tags: list[str] | None = None,
        description: str = "",
        source_file: str = "",
    ) -> Snippet:
        """Save a new snippet."""
        now = time.time()
        snippet = Snippet(
            id=self._generate_id(),
            title=title,
            code=code,
            language=language,
            tags=tags or [],
            description=description,
            source_file=source_file,
            created_at=now,
            updated_at=now,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO snippets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snippet.id,
                    snippet.title,
                    snippet.code,
                    snippet.language,
                    json.dumps(snippet.tags),
                    snippet.description,
                    snippet.source_file,
                    snippet.created_at,
                    snippet.updated_at,
                    snippet.usage_count,
                ),
            )
            # Index for search
            search_content = (
                f"{snippet.title} {snippet.description} {' '.join(snippet.tags)} {snippet.code}"
            )
            conn.execute(
                "INSERT OR REPLACE INTO snippet_fts VALUES (?, ?)",
                (snippet.id, search_content.lower()),
            )
            conn.commit()

        return snippet

    def search(
        self, query: str, language: str | None = None, tag: str | None = None, limit: int = 20
    ) -> list[Snippet]:
        """Search snippets by query, language, or tag."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if query:
                # Search in FTS table
                query_lower = query.lower()
                rows = conn.execute(
                    """SELECT s.* FROM snippets s
                       JOIN snippet_fts f ON s.id = f.id
                       WHERE f.content LIKE ?
                       ORDER BY s.usage_count DESC, s.updated_at DESC
                       LIMIT ?""",
                    (f"%{query_lower}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM snippets ORDER BY usage_count DESC, updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            snippets = [self._row_to_snippet(r) for r in rows]

            if language:
                snippets = [s for s in snippets if s.language.lower() == language.lower()]
            if tag:
                snippets = [s for s in snippets if tag.lower() in [t.lower() for t in s.tags]]

            return snippets

    def get(self, snippet_id: str) -> Snippet | None:
        """Get a snippet by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM snippets WHERE id = ?", (snippet_id,)).fetchone()
            if row:
                # Increment usage
                conn.execute(
                    "UPDATE snippets SET usage_count = usage_count + 1 WHERE id = ?",
                    (snippet_id,),
                )
                conn.commit()
                return self._row_to_snippet(row)
        return None

    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM snippets WHERE id = ?", (snippet_id,))
            conn.execute("DELETE FROM snippet_fts WHERE id = ?", (snippet_id,))
            conn.commit()
            return conn.total_changes > 0

    def list_tags(self) -> dict[str, int]:
        """List all tags with counts."""
        tags: dict[str, int] = {}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT tags FROM snippets").fetchall()
            for row in rows:
                for tag in json.loads(row[0]):
                    tags[tag] = tags.get(tag, 0) + 1
        return dict(sorted(tags.items(), key=lambda x: -x[1]))

    def list_languages(self) -> dict[str, int]:
        """List all languages with counts."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT language, COUNT(*) FROM snippets WHERE language != '' GROUP BY language ORDER BY COUNT(*) DESC"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def count(self) -> int:
        """Total snippet count."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]

    def export_all(self) -> list[dict]:
        """Export all snippets as dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM snippets ORDER BY created_at DESC").fetchall()
            return [asdict(self._row_to_snippet(r)) for r in rows]

    def _row_to_snippet(self, row) -> Snippet:
        return Snippet(
            id=row["id"],
            title=row["title"],
            code=row["code"],
            language=row["language"],
            tags=json.loads(row["tags"]),
            description=row["description"],
            source_file=row["source_file"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            usage_count=row["usage_count"],
        )
