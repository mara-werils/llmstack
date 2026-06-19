"""Persistent index — SQLite-backed file index with incremental updates.

Stores file hashes, chunks, and embeddings on disk. Only re-embeds files
that have changed since the last indexing run. Turns 30s re-index into ~0.1s.

Index location: .llmstack-index/ in the project root.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llmstack.ask.parsers import TextChunk

logger = logging.getLogger(__name__)

INDEX_DIR = ".llmstack-index"
DB_NAME = "index.db"
EMBEDDINGS_NAME = "embeddings.npy"
SCHEMA_VERSION = 1


@dataclass
class IndexStats:
    """Statistics about an indexing operation."""

    total_files: int = 0
    unchanged: int = 0
    added: int = 0
    modified: int = 0
    removed: int = 0
    total_chunks: int = 0
    index_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "unchanged": self.unchanged,
            "added": self.added,
            "modified": self.modified,
            "removed": self.removed,
            "total_chunks": self.total_chunks,
            "index_time_ms": round(self.index_time_ms, 1),
        }


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


class PersistentIndex:
    """SQLite-backed persistent index for file chunks and embeddings.

    Usage:
        index = PersistentIndex("/path/to/project")
        changed_files = index.diff(current_files)
        # ... parse and embed only changed files ...
        index.update(new_chunks, new_embeddings, file_hashes)
        chunks, embeddings = index.load()
    """

    def __init__(self, project_dir: str | Path):
        self._project_dir = Path(project_dir).resolve()
        self._index_dir = self._project_dir / INDEX_DIR
        self._db_path = self._index_dir / DB_NAME
        self._emb_path = self._index_dir / EMBEDDINGS_NAME
        self._conn: sqlite3.Connection | None = None

    def _ensure_db(self) -> sqlite3.Connection:
        """Create index directory and database if needed."""
        if self._conn is not None:
            return self._conn

        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Enable FK enforcement so the chunks->files ON DELETE CASCADE below
        # actually fires (SQLite defaults foreign_keys to OFF per-connection).
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
            );
        """)

        # Check schema version
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

        return self._conn

    def exists(self) -> bool:
        """Check if a persistent index already exists."""
        return self._db_path.is_file()

    def diff(self, current_files: list[Path]) -> tuple[list[Path], list[Path], list[str]]:
        """Compare current files against the stored index.

        Returns (files_to_add_or_update, new_files_list, removed_paths).
        """
        conn = self._ensure_db()

        # Get stored file hashes
        stored: dict[str, str] = {}
        for row in conn.execute("SELECT path, hash FROM files"):
            stored[row[0]] = row[1]

        to_update: list[Path] = []
        current_paths: set[str] = set()

        for fpath in current_files:
            rel = self._rel_path(fpath)
            current_paths.add(rel)
            current_hash = _file_hash(fpath)

            if rel not in stored:
                to_update.append(fpath)
            elif stored[rel] != current_hash:
                to_update.append(fpath)

        # Find removed files
        removed = [p for p in stored if p not in current_paths]

        return to_update, current_files, removed

    def update(
        self,
        file_chunks: dict[str, list[TextChunk]],
        embeddings_array: np.ndarray | None,
        file_hashes: dict[str, str],
        all_chunks: list[TextChunk],
    ) -> None:
        """Update the index with new/modified file data.

        Args:
            file_chunks: mapping of relative file path → chunks for changed files
            embeddings_array: full embeddings array for ALL chunks (not just changed)
            file_hashes: mapping of relative file path → SHA-256 hash
            all_chunks: complete ordered list of all chunks (matches embeddings_array)
        """
        conn = self._ensure_db()

        # Remove old data for changed files
        for rel_path in file_chunks:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
            conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))

        # Insert new file records
        for rel_path, fhash in file_hashes.items():
            conn.execute(
                "INSERT OR REPLACE INTO files (path, hash, mtime) VALUES (?, ?, ?)",
                (rel_path, fhash, time.time()),
            )

        # Insert chunks
        for rel_path, chunks in file_chunks.items():
            for i, chunk in enumerate(chunks):
                conn.execute(
                    "INSERT INTO chunks (file_path, content, start_line, end_line, chunk_index) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rel_path, chunk.content, chunk.start_line, chunk.end_line, i),
                )

        conn.commit()

        # Save embeddings
        if embeddings_array is not None and embeddings_array.size > 0:
            np.save(str(self._emb_path), embeddings_array)

        # Save chunk order metadata
        chunk_meta = [
            {"source": c.source, "start_line": c.start_line, "end_line": c.end_line}
            for c in all_chunks
        ]
        meta_path = self._index_dir / "chunks_order.json"
        meta_path.write_text(json.dumps(chunk_meta))

        logger.info("Index updated: %d files, %d total chunks", len(file_hashes), len(all_chunks))

    def remove_files(self, rel_paths: list[str]) -> None:
        """Remove files and their chunks from the index."""
        conn = self._ensure_db()
        for rel_path in rel_paths:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
            conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))
        conn.commit()

    def load_chunks(self) -> list[TextChunk]:
        """Load all chunks from the index in order."""
        conn = self._ensure_db()
        chunks: list[TextChunk] = []

        rows = conn.execute(
            "SELECT file_path, content, start_line, end_line FROM chunks "
            "ORDER BY file_path, chunk_index"
        ).fetchall()

        for row in rows:
            chunks.append(
                TextChunk(
                    content=row[1],
                    source=row[0],
                    start_line=row[2],
                    end_line=row[3],
                )
            )
        return chunks

    def load_embeddings(self) -> np.ndarray | None:
        """Load stored embeddings from disk."""
        if not self._emb_path.is_file():
            return None
        try:
            return np.load(str(self._emb_path))
        except Exception:
            return None

    def file_count(self) -> int:
        """Return the number of indexed files."""
        conn = self._ensure_db()
        row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return row[0] if row else 0

    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        conn = self._ensure_db()
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        """Clear the entire index."""
        conn = self._ensure_db()
        conn.executescript("""
            DELETE FROM chunks;
            DELETE FROM files;
        """)
        conn.commit()
        if self._emb_path.is_file():
            self._emb_path.unlink()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _rel_path(self, fpath: Path) -> str:
        """Get path relative to project dir."""
        try:
            return str(fpath.resolve().relative_to(self._project_dir))
        except ValueError:
            return str(fpath)
