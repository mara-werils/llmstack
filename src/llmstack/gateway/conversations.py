"""Conversation history persistence — SQLite-backed storage for chat sessions.

Provides durable conversation storage with search, pagination, and metadata
tracking. Each conversation can span multiple messages and models.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_DB_PATH = Path.home() / ".llmstack" / "conversations.db"


@dataclass
class Message:
    """A single message in a conversation."""

    role: str
    content: str
    model: str = ""
    timestamp: float = 0.0
    tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "model": self.model,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Conversation:
    """A conversation session with messages."""

    id: str = ""
    title: str = ""
    model: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    message_count: int = 0
    total_tokens: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def age_days(self) -> float:
        """Return how many days ago this conversation was created."""
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "tags": self.tags,
        }


class ConversationStore:
    """SQLite-backed conversation storage with full-text search."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        self._lock = Lock()
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at DESC);
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_conversation(
        self,
        title: str = "",
        model: str = "",
        tags: list[str] | None = None,
    ) -> Conversation:
        """Start a new conversation."""
        conv = Conversation(title=title, model=model, tags=tags or [])
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO conversations
                   (id, title, model, created_at, updated_at, tags)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    conv.id,
                    conv.title,
                    conv.model,
                    conv.created_at,
                    conv.updated_at,
                    json.dumps(conv.tags),
                ),
            )
        return conv

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: str = "",
        tokens: int = 0,
        latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        msg = Message(
            role=role,
            content=content,
            model=model,
            tokens=tokens,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, model, timestamp, tokens, latency_ms, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    msg.role,
                    msg.content,
                    msg.model,
                    msg.timestamp,
                    msg.tokens,
                    msg.latency_ms,
                    json.dumps(msg.metadata),
                ),
            )
            conn.execute(
                """UPDATE conversations SET
                   message_count = message_count + 1,
                   total_tokens = total_tokens + ?,
                   updated_at = ?
                   WHERE id = ?""",
                (tokens, time.time(), conversation_id),
            )
        return msg

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get conversation metadata."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_conversation(row)

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        """Get messages for a conversation."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM messages
                   WHERE conversation_id = ?
                   ORDER BY timestamp ASC
                   LIMIT ? OFFSET ?""",
                (conversation_id, limit, offset),
            ).fetchall()
            return [self._row_to_message(r) for r in rows]

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> list[Conversation]:
        """List recent conversations with optional search."""
        with self._connect() as conn:
            if search:
                rows = conn.execute(
                    """SELECT DISTINCT c.* FROM conversations c
                       LEFT JOIN messages m ON c.id = m.conversation_id
                       WHERE c.title LIKE ? OR m.content LIKE ?
                       ORDER BY c.updated_at DESC
                       LIMIT ? OFFSET ?""",
                    (f"%{search}%", f"%{search}%", limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM conversations
                       ORDER BY updated_at DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()
            return [self._row_to_conversation(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            result = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            return result.rowcount > 0

    def get_stats(self) -> dict:
        """Get conversation statistics."""
        with self._connect() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_tokens = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM conversations"
            ).fetchone()[0]
            return {
                "total_conversations": conv_count,
                "total_messages": msg_count,
                "total_tokens": total_tokens,
            }

    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            total_tokens=row["total_tokens"],
            tags=json.loads(row["tags"]),
        )

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            role=row["role"],
            content=row["content"],
            model=row["model"],
            timestamp=row["timestamp"],
            tokens=row["tokens"],
            latency_ms=row["latency_ms"],
            metadata=json.loads(row["metadata"]),
        )
