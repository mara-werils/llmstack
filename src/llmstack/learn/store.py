"""Feedback storage — persistent SQLite store for all learning signals.

Stores feedback events, training history, and model performance metrics.
Uses SQLite for zero-config, local-first persistence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from llmstack.learn.feedback import Feedback, FeedbackType

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".llmstack" / "learning.db"

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    feedback_type TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT DEFAULT '',
    provider TEXT DEFAULT '',
    correction TEXT DEFAULT '',
    edit_diff TEXT DEFAULT '',
    preferred_over TEXT DEFAULT '',
    rating INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    command TEXT DEFAULT '',
    context TEXT DEFAULT '{}',
    used_in_training INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS train_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    model_version TEXT NOT NULL,
    base_model TEXT NOT NULL,
    feedback_count INTEGER NOT NULL,
    dataset_size INTEGER NOT NULL,
    final_loss REAL,
    best_loss REAL,
    train_time_seconds REAL,
    status TEXT DEFAULT 'completed',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    base_model TEXT NOT NULL,
    adapter_path TEXT DEFAULT '',
    train_run_id INTEGER,
    quality_score REAL DEFAULT 0.0,
    is_active INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS quality_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    model_version TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    sample_size INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model);
CREATE INDEX IF NOT EXISTS idx_feedback_unused ON feedback(used_in_training) WHERE used_in_training = 0;
CREATE INDEX IF NOT EXISTS idx_quality_version ON quality_snapshots(model_version, timestamp);
"""


class FeedbackStore:
    """SQLite-backed store for feedback and learning state."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    @property
    def is_connected(self) -> bool:
        """Return True if a database connection is currently open."""
        return self._conn is not None

    @property
    def db_size_bytes(self) -> int:
        """Return the size of the database file in bytes."""
        return self.db_path.stat().st_size if self.db_path.exists() else 0

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            ("version", str(_SCHEMA_VERSION)),
        )
        self.conn.commit()

    def add_feedback(self, feedback: Feedback) -> None:
        """Store a feedback event."""
        self.conn.execute(
            """INSERT OR REPLACE INTO feedback
            (id, timestamp, feedback_type, query, response, model, provider,
             correction, edit_diff, preferred_over, rating, tags, command, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                feedback.id,
                feedback.timestamp,
                feedback.feedback_type.value,
                feedback.query,
                feedback.response,
                feedback.model,
                feedback.provider,
                feedback.correction,
                feedback.edit_diff,
                feedback.preferred_over,
                feedback.rating,
                json.dumps(feedback.tags),
                feedback.command,
                json.dumps(feedback.context),
            ),
        )
        self.conn.commit()

    def get_feedback(
        self,
        feedback_type: FeedbackType | None = None,
        model: str | None = None,
        since: float | None = None,
        unused_only: bool = False,
        limit: int = 1000,
    ) -> list[Feedback]:
        """Query feedback with optional filters."""
        conditions = []
        params: list[Any] = []

        if feedback_type:
            conditions.append("feedback_type = ?")
            params.append(feedback_type.value)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if unused_only:
            conditions.append("used_in_training = 0")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM feedback {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_feedback(row) for row in rows]

    def get_unused_feedback_count(self) -> int:
        """Count feedback not yet used in training."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM feedback WHERE used_in_training = 0"
        ).fetchone()
        return row["cnt"] if row else 0

    def mark_feedback_used(self, feedback_ids: list[str]) -> None:
        """Mark feedback as used in training."""
        if not feedback_ids:
            return
        placeholders = ",".join("?" * len(feedback_ids))
        self.conn.execute(
            f"UPDATE feedback SET used_in_training = 1 WHERE id IN ({placeholders})",
            feedback_ids,
        )
        self.conn.commit()

    def add_train_run(
        self,
        model_version: str,
        base_model: str,
        feedback_count: int,
        dataset_size: int,
        final_loss: float = 0.0,
        best_loss: float = 0.0,
        train_time_seconds: float = 0.0,
        status: str = "completed",
        metadata: dict | None = None,
    ) -> int:
        """Record a training run."""
        cursor = self.conn.execute(
            """INSERT INTO train_runs
            (timestamp, model_version, base_model, feedback_count, dataset_size,
             final_loss, best_loss, train_time_seconds, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                model_version,
                base_model,
                feedback_count,
                dataset_size,
                final_loss,
                best_loss,
                train_time_seconds,
                status,
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def add_model_version(
        self,
        version: str,
        base_model: str,
        adapter_path: str = "",
        train_run_id: int = 0,
        quality_score: float = 0.0,
        is_active: bool = False,
        metadata: dict | None = None,
    ) -> None:
        """Register a model version."""
        if is_active:
            self.conn.execute("UPDATE model_versions SET is_active = 0")
        self.conn.execute(
            """INSERT OR REPLACE INTO model_versions
            (version, timestamp, base_model, adapter_path, train_run_id,
             quality_score, is_active, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version,
                time.time(),
                base_model,
                adapter_path,
                train_run_id,
                quality_score,
                1 if is_active else 0,
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()

    def get_active_version(self) -> dict[str, Any] | None:
        """Get the currently active model version."""
        row = self.conn.execute("SELECT * FROM model_versions WHERE is_active = 1").fetchone()
        return dict(row) if row else None

    def get_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List model versions ordered by recency."""
        rows = self.conn.execute(
            "SELECT * FROM model_versions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def add_quality_snapshot(
        self,
        model_version: str,
        metric: str,
        value: float,
        sample_size: int = 0,
    ) -> None:
        """Record a quality metric measurement."""
        self.conn.execute(
            """INSERT INTO quality_snapshots
            (timestamp, model_version, metric, value, sample_size)
            VALUES (?, ?, ?, ?, ?)""",
            (time.time(), model_version, metric, value, sample_size),
        )
        self.conn.commit()

    def get_quality_trend(
        self, model_version: str, metric: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get quality metric trend for a version."""
        rows = self.conn.execute(
            """SELECT timestamp, value, sample_size FROM quality_snapshots
            WHERE model_version = ? AND metric = ?
            ORDER BY timestamp DESC LIMIT ?""",
            (model_version, metric, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        total = self.conn.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()
        unused = self.get_unused_feedback_count()
        by_type = self.conn.execute(
            "SELECT feedback_type, COUNT(*) as cnt FROM feedback GROUP BY feedback_type"
        ).fetchall()
        runs = self.conn.execute("SELECT COUNT(*) as cnt FROM train_runs").fetchone()
        versions = self.conn.execute("SELECT COUNT(*) as cnt FROM model_versions").fetchone()

        return {
            "total_feedback": total["cnt"] if total else 0,
            "unused_feedback": unused,
            "feedback_by_type": {row["feedback_type"]: row["cnt"] for row in by_type},
            "total_train_runs": runs["cnt"] if runs else 0,
            "total_versions": versions["cnt"] if versions else 0,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_feedback(self, row: sqlite3.Row) -> Feedback:
        """Convert a database row to a Feedback object."""
        data = dict(row)
        data["tags"] = json.loads(data.get("tags", "[]"))
        data["context"] = json.loads(data.get("context", "{}"))
        data.pop("used_in_training", None)
        return Feedback.from_dict(data)
