"""llmstack bookmarks — Save and manage important conversation snippets."""

from __future__ import annotations

import json
import sqlite3
import time
import hashlib
from pathlib import Path

from llmstack.cli.console import console


class BookmarkManager:
    """Manage conversation bookmarks."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            data_dir = Path.home() / ".llmstack" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "bookmarks.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def add(self, title: str, content: str, category: str = "general",
            tags: list[str] | None = None, source: str = "", notes: str = "") -> str:
        bookmark_id = hashlib.sha256(f"{title}{time.time()}".encode()).hexdigest()[:10]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO bookmarks (id, title, content, category, tags, source, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (bookmark_id, title, content, category, json.dumps(tags or []),
                 source, notes, time.time()),
            )
            conn.commit()
        return bookmark_id

    def list_all(self, category: str | None = None, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if category:
                rows = conn.execute(
                    "SELECT * FROM bookmarks WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM bookmarks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get(self, bookmark_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM bookmarks WHERE id = ?", (bookmark_id,)).fetchone()
            return dict(row) if row else None

    def search(self, query: str, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM bookmarks WHERE title LIKE ? OR content LIKE ? OR notes LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, bookmark_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
            conn.commit()
            return conn.total_changes > 0

    def categories(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT category, COUNT(*) FROM bookmarks GROUP BY category ORDER BY COUNT(*) DESC"
            ).fetchall()
            return {r[0]: r[1] for r in rows}


def bookmarks(
    action: str = "list",
    query: str = "",
    title: str | None = None,
    content: str | None = None,
    category: str = "general",
    tags: str | None = None,
    notes: str = "",
    limit: int = 50,
) -> None:
    """Manage bookmarks."""
    from rich.table import Table
    from rich.panel import Panel
    from datetime import datetime

    mgr = BookmarkManager()

    if action == "add":
        if not title or not content:
            console.print("[error]Provide --title and --content for add[/]")
            return
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        bid = mgr.add(title=title, content=content, category=category,
                       tags=tag_list, notes=notes)
        console.print(f"[green]Bookmark saved:[/] [bold]{title}[/]  id=[dim]{bid}[/]")

    elif action == "list":
        items = mgr.list_all(category=category if category != "general" else None, limit=limit)
        if not items:
            console.print("[dim]No bookmarks yet. Save one with: llmstack bookmarks add --title '...' --content '...'[/]")
            return

        table = Table(title="Bookmarks", show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("ID", width=10)
        table.add_column("Title", style="bold")
        table.add_column("Category", width=12)
        table.add_column("Date", width=12)
        table.add_column("Preview", style="dim")

        for item in items:
            dt = datetime.fromtimestamp(item["created_at"]).strftime("%Y-%m-%d")
            preview = item["content"][:60].replace("\n", " ")
            table.add_row(item["id"], item["title"], item["category"], dt, preview)

        console.print(table)

    elif action == "show":
        if not query:
            console.print("[error]Provide bookmark ID[/]")
            return
        item = mgr.get(query)
        if not item:
            console.print(f"[error]Bookmark not found: {query}[/]")
            return

        tags_str = ", ".join(json.loads(item.get("tags", "[]")))
        console.print()
        console.print(f"[bold]{item['title']}[/]  category=[dim]{item['category']}[/]")
        if tags_str:
            console.print(f"  Tags: {tags_str}")
        if item.get("notes"):
            console.print(f"  Notes: [dim]{item['notes']}[/]")
        console.print()
        console.print(Panel(item["content"], border_style="cyan"))

    elif action == "search":
        results = mgr.search(query, limit=limit)
        if not results:
            console.print(f"[dim]No bookmarks matching '{query}'[/]")
            return
        for item in results:
            preview = item["content"][:80].replace("\n", " ")
            console.print(f"  [bold]{item['id']}[/] {item['title']}  [dim]{preview}[/]")

    elif action == "delete":
        if mgr.delete(query):
            console.print(f"[green]Bookmark {query} deleted.[/]")
        else:
            console.print(f"[error]Bookmark not found: {query}[/]")

    elif action == "categories":
        cats = mgr.categories()
        if not cats:
            console.print("[dim]No categories yet.[/]")
            return
        for cat, count in cats.items():
            console.print(f"  {cat}: {count}")
