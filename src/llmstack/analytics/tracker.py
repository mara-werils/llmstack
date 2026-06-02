"""Usage analytics tracker — track command usage, model performance, and trends."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UsageEvent:
    """A single usage event."""
    command: str
    model: str
    tokens_in: int
    tokens_out: int
    duration: float
    success: bool
    timestamp: float


class AnalyticsTracker:
    """Track and analyze llmstack usage patterns."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            data_dir = Path.home() / ".llmstack" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "analytics.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    duration REAL DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_cmd ON events(command)
            """)
            conn.commit()

    def track(self, event: UsageEvent) -> None:
        """Record a usage event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events (command, model, tokens_in, tokens_out, duration, success, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.command, event.model, event.tokens_in, event.tokens_out,
                 event.duration, int(event.success), event.timestamp),
            )
            conn.commit()

    def get_summary(self, days: int = 30) -> dict:
        """Get usage summary for the last N days."""
        cutoff = time.time() - (days * 86400)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total stats
            row = conn.execute(
                """SELECT COUNT(*) as total, SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out, SUM(duration) as duration,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                   FROM events WHERE timestamp > ?""",
                (cutoff,),
            ).fetchone()

            total = row["total"] or 0
            tokens_in = row["tokens_in"] or 0
            tokens_out = row["tokens_out"] or 0
            total_duration = row["duration"] or 0
            successes = row["successes"] or 0

            # By command
            cmd_rows = conn.execute(
                """SELECT command, COUNT(*) as count, SUM(duration) as duration,
                   SUM(tokens_in + tokens_out) as tokens
                   FROM events WHERE timestamp > ?
                   GROUP BY command ORDER BY count DESC""",
                (cutoff,),
            ).fetchall()

            # By model
            model_rows = conn.execute(
                """SELECT model, COUNT(*) as count, AVG(duration) as avg_duration,
                   SUM(tokens_in + tokens_out) as tokens
                   FROM events WHERE timestamp > ? AND model != ''
                   GROUP BY model ORDER BY count DESC""",
                (cutoff,),
            ).fetchall()

            # Daily trend
            daily_rows = conn.execute(
                """SELECT date(timestamp, 'unixepoch') as day, COUNT(*) as count
                   FROM events WHERE timestamp > ?
                   GROUP BY day ORDER BY day""",
                (cutoff,),
            ).fetchall()

            # Hourly distribution
            hourly_rows = conn.execute(
                """SELECT CAST(strftime('%H', timestamp, 'unixepoch') AS INTEGER) as hour,
                   COUNT(*) as count FROM events WHERE timestamp > ?
                   GROUP BY hour ORDER BY hour""",
                (cutoff,),
            ).fetchall()

            return {
                "period_days": days,
                "total_requests": total,
                "total_tokens": tokens_in + tokens_out,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "total_duration": total_duration,
                "avg_duration": total_duration / max(1, total),
                "success_rate": successes / max(1, total) * 100,
                "by_command": [dict(r) for r in cmd_rows],
                "by_model": [dict(r) for r in model_rows],
                "daily_trend": [dict(r) for r in daily_rows],
                "hourly_dist": [dict(r) for r in hourly_rows],
            }

    def get_streak(self) -> int:
        """Get current usage streak in days."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT DISTINCT date(timestamp, 'unixepoch') as day
                   FROM events ORDER BY day DESC LIMIT 365"""
            ).fetchall()

        if not rows:
            return 0

        from datetime import date, timedelta
        today = date.today()
        streak = 0

        for i, row in enumerate(rows):
            expected = today - timedelta(days=i)
            if row[0] == expected.isoformat():
                streak += 1
            else:
                break

        return streak
