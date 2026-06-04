"""Plugin registry — discover, install, and manage llmstack plugins."""

from __future__ import annotations

import json
import importlib
import importlib.metadata
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PluginInfo:
    """Information about a plugin."""

    name: str
    version: str
    description: str
    author: str
    entry_point: str
    plugin_type: str  # command, provider, middleware, tool
    installed: bool = False
    enabled: bool = True
    config: dict | None = None


# Built-in plugin types and their entry points
PLUGIN_TYPES = {
    "command": "llmstack.commands",
    "provider": "llmstack.providers",
    "middleware": "llmstack.middleware",
    "tool": "llmstack.tools",
    "formatter": "llmstack.formatters",
}


class PluginRegistry:
    """Manage plugins via entry_points and local registration."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            data_dir = Path.home() / ".llmstack" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "plugins.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plugins (
                    name TEXT PRIMARY KEY,
                    version TEXT DEFAULT '0.0.0',
                    description TEXT DEFAULT '',
                    author TEXT DEFAULT '',
                    entry_point TEXT DEFAULT '',
                    plugin_type TEXT DEFAULT 'command',
                    enabled INTEGER DEFAULT 1,
                    config TEXT DEFAULT '{}',
                    installed_at REAL NOT NULL
                )
            """)
            conn.commit()

    def discover(self) -> list[PluginInfo]:
        """Discover installed plugins via entry_points."""
        plugins = []

        for plugin_type, group in PLUGIN_TYPES.items():
            try:
                eps = importlib.metadata.entry_points()
                # Python 3.12+ returns SelectableGroups
                if hasattr(eps, "select"):
                    group_eps = eps.select(group=group)
                else:
                    group_eps = eps.get(group, [])

                for ep in group_eps:
                    plugins.append(
                        PluginInfo(
                            name=ep.name,
                            version=getattr(ep.dist, "version", "0.0.0") if ep.dist else "0.0.0",
                            description=f"Plugin: {ep.name}",
                            author="",
                            entry_point=str(ep),
                            plugin_type=plugin_type,
                            installed=True,
                        )
                    )
            except Exception:
                pass

        # Add locally registered plugins
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT * FROM plugins").fetchall():
                plugins.append(
                    PluginInfo(
                        name=row["name"],
                        version=row["version"],
                        description=row["description"],
                        author=row["author"],
                        entry_point=row["entry_point"],
                        plugin_type=row["plugin_type"],
                        installed=True,
                        enabled=bool(row["enabled"]),
                        config=json.loads(row["config"]),
                    )
                )

        return plugins

    def register(self, plugin: PluginInfo) -> None:
        """Register a plugin locally."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO plugins
                   (name, version, description, author, entry_point, plugin_type, enabled, config, installed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    plugin.name,
                    plugin.version,
                    plugin.description,
                    plugin.author,
                    plugin.entry_point,
                    plugin.plugin_type,
                    int(plugin.enabled),
                    json.dumps(plugin.config or {}),
                    time.time(),
                ),
            )
            conn.commit()

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM plugins WHERE name = ?", (name,))
            conn.commit()
            return conn.total_changes > 0

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE plugins SET enabled = 1 WHERE name = ?", (name,))
            conn.commit()
            return conn.total_changes > 0

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE plugins SET enabled = 0 WHERE name = ?", (name,))
            conn.commit()
            return conn.total_changes > 0

    def load_plugin(self, name: str) -> object | None:
        """Load and instantiate a plugin."""
        plugins = self.discover()
        for p in plugins:
            if p.name == name and p.enabled:
                try:
                    ep_str = p.entry_point
                    if ":" in ep_str:
                        module_path, attr = ep_str.rsplit(":", 1)
                        module = importlib.import_module(module_path)
                        return getattr(module, attr)
                    else:
                        return importlib.import_module(ep_str)
                except Exception:
                    return None
        return None

    def get_enabled(self, plugin_type: str | None = None) -> list[PluginInfo]:
        """Get all enabled plugins, optionally filtered by type."""
        plugins = self.discover()
        result = [p for p in plugins if p.enabled]
        if plugin_type:
            result = [p for p in result if p.plugin_type == plugin_type]
        return result
