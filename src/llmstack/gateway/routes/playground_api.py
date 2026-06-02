"""Enhanced playground API routes — comparison, sharing, history."""

from __future__ import annotations

import json
import sqlite3
import time
import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/playground", tags=["playground"])


class PlaygroundSession(BaseModel):
    """A playground session with messages and settings."""
    id: str = ""
    title: str = ""
    model: str = "llama3.2"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    messages: list[dict] = Field(default_factory=list)
    created_at: float = 0
    updated_at: float = 0


class CompareRequest(BaseModel):
    """Request to compare multiple models."""
    prompt: str
    models: list[str]
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


class ShareRequest(BaseModel):
    """Request to create a shareable conversation."""
    session_id: str
    title: str = ""


class PlaygroundStore:
    """SQLite-backed playground session storage."""

    def __init__(self):
        data_dir = Path.home() / ".llmstack" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "playground.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    model TEXT DEFAULT 'llama3.2',
                    system_prompt TEXT DEFAULT '',
                    temperature REAL DEFAULT 0.7,
                    max_tokens INTEGER DEFAULT 2048,
                    messages TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shared (
                    share_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def save_session(self, session: PlaygroundSession) -> PlaygroundSession:
        now = time.time()
        if not session.id:
            session.id = hashlib.sha256(str(now).encode()).hexdigest()[:12]
            session.created_at = now
        session.updated_at = now

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, title, model, system_prompt, temperature, max_tokens, messages, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session.id, session.title, session.model, session.system_prompt,
                 session.temperature, session.max_tokens,
                 json.dumps(session.messages), session.created_at, session.updated_at),
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> PlaygroundSession | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                return PlaygroundSession(
                    id=row["id"], title=row["title"], model=row["model"],
                    system_prompt=row["system_prompt"], temperature=row["temperature"],
                    max_tokens=row["max_tokens"], messages=json.loads(row["messages"]),
                    created_at=row["created_at"], updated_at=row["updated_at"],
                )
        return None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, title, model, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return conn.total_changes > 0

    def share_session(self, session_id: str, title: str = "") -> str:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        share_id = hashlib.sha256(f"{session_id}{time.time()}".encode()).hexdigest()[:16]
        data = json.dumps({
            "title": title or session.title,
            "model": session.model,
            "messages": session.messages,
        })

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO shared (share_id, session_id, title, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (share_id, session_id, title, data, time.time()),
            )
            conn.commit()
        return share_id

    def get_shared(self, share_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM shared WHERE share_id = ?", (share_id,)).fetchone()
            if row:
                return {"share_id": row["share_id"], **json.loads(row["data"])}
        return None


_store = PlaygroundStore()


@router.get("/sessions")
async def list_sessions(limit: int = 20):
    """List saved playground sessions."""
    return {"sessions": _store.list_sessions(limit)}


@router.post("/sessions")
async def save_session(session: PlaygroundSession):
    """Save a playground session."""
    saved = _store.save_session(session)
    return {"id": saved.id, "status": "saved"}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session."""
    session = _store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if _store.delete_session(session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/share")
async def share_session(req: ShareRequest):
    """Create a shareable link for a session."""
    try:
        share_id = _store.share_session(req.session_id, req.title)
        return {"share_id": share_id, "status": "shared"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/shared/{share_id}")
async def get_shared(share_id: str):
    """Get a shared conversation."""
    data = _store.get_shared(share_id)
    if not data:
        raise HTTPException(status_code=404, detail="Shared session not found")
    return data
