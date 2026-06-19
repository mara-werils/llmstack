"""Tests for the plugins package: registry, loader, spec, and __init__."""

from __future__ import annotations

import json
import sqlite3
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the package graph in dependency order first: importing
# llmstack.plugins.loader / .spec as the very first import triggers a
# pre-existing circular import (services.registry -> core.stack -> services).
# Priming llmstack.core (as the existing test suite does) resolves it.
import llmstack.core  # noqa: F401  (import for side effect / ordering)
import llmstack.plugins as plugins_pkg
from llmstack.plugins import PluginInfo, PluginRegistry
from llmstack.plugins.loader import ServiceRegistry
from llmstack.plugins.registry import (
    PLUGIN_TYPES,
    PluginInfo as RegistryPluginInfo,
    PluginRegistry as RegistryPluginRegistry,
)
from llmstack.plugins.spec import ServiceBase
from llmstack.services.base import ServiceBase as RealServiceBase
from llmstack.services.registry import ServiceRegistry as RealServiceRegistry


# ── fixtures / helpers ─────────────────────────────────────────────


@pytest.fixture
def registry(tmp_path: Path) -> PluginRegistry:
    """A registry backed by a temporary sqlite db."""
    return PluginRegistry(db_path=tmp_path / "plugins.db")


def _plugin(name: str = "demo", **overrides) -> PluginInfo:
    defaults = dict(
        name=name,
        version="1.2.3",
        description="A demo plugin",
        author="alice",
        entry_point="demo_pkg:DemoPlugin",
        plugin_type="command",
    )
    defaults.update(overrides)
    return PluginInfo(**defaults)


def _no_entry_points():
    """Patch importlib.metadata.entry_points to return an empty container.

    Keeps discover() from picking up whatever is installed in the test env,
    so locally-registered plugins are the only thing under test.
    """
    fake = types.SimpleNamespace(select=lambda group: [])
    return patch("importlib.metadata.entry_points", return_value=fake)


# ── package re-exports ─────────────────────────────────────────────


def test_package_reexports_identity():
    assert plugins_pkg.PluginInfo is RegistryPluginInfo
    assert plugins_pkg.PluginRegistry is RegistryPluginRegistry
    assert PluginInfo is RegistryPluginInfo
    assert PluginRegistry is RegistryPluginRegistry


def test_package_all():
    assert set(plugins_pkg.__all__) == {"PluginInfo", "PluginRegistry"}


def test_loader_reexports_service_registry():
    assert ServiceRegistry is RealServiceRegistry
    from llmstack.plugins import loader

    assert loader.__all__ == ["ServiceRegistry"]


def test_spec_reexports_service_base():
    assert ServiceBase is RealServiceBase
    from llmstack.plugins import spec

    assert spec.__all__ == ["ServiceBase"]


# ── PluginInfo dataclass ───────────────────────────────────────────


def test_plugin_info_defaults():
    info = PluginInfo(
        name="x",
        version="0.1",
        description="d",
        author="a",
        entry_point="e",
        plugin_type="command",
    )
    assert info.installed is False
    assert info.enabled is True
    assert info.config is None


def test_plugin_info_explicit_fields():
    info = _plugin(installed=True, enabled=False, config={"k": "v"})
    assert info.installed is True
    assert info.enabled is False
    assert info.config == {"k": "v"}


def test_plugin_types_mapping():
    assert PLUGIN_TYPES["command"] == "llmstack.commands"
    assert PLUGIN_TYPES["provider"] == "llmstack.providers"
    assert PLUGIN_TYPES["middleware"] == "llmstack.middleware"
    assert PLUGIN_TYPES["tool"] == "llmstack.tools"
    assert PLUGIN_TYPES["formatter"] == "llmstack.formatters"


# ── construction / db init ─────────────────────────────────────────


def test_init_creates_db_and_table(tmp_path: Path):
    db = tmp_path / "p.db"
    PluginRegistry(db_path=db)
    assert db.exists()
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='plugins'"
        ).fetchall()
    assert rows == [("plugins",)]


def test_init_accepts_string_path(tmp_path: Path):
    db = tmp_path / "str.db"
    reg = PluginRegistry(db_path=str(db))
    assert isinstance(reg.db_path, Path)
    assert reg.db_path == db


def test_init_default_path_under_home(tmp_path: Path):
    fake_home = tmp_path / "home"
    with patch.object(Path, "home", return_value=fake_home):
        reg = PluginRegistry()
    expected = fake_home / ".llmstack" / "data" / "plugins.db"
    assert reg.db_path == expected
    assert expected.exists()


def test_init_idempotent(tmp_path: Path):
    db = tmp_path / "idem.db"
    PluginRegistry(db_path=db)
    # Re-init over the same DB should not raise (CREATE TABLE IF NOT EXISTS).
    reg2 = PluginRegistry(db_path=db)
    assert reg2.registered_count == 0


# ── register / registered_count / is_registered ────────────────────


def test_register_and_count(registry: PluginRegistry):
    assert registry.registered_count == 0
    registry.register(_plugin("alpha"))
    assert registry.registered_count == 1
    assert registry.is_registered("alpha")
    assert not registry.is_registered("missing")


def test_register_replace_does_not_duplicate(registry: PluginRegistry):
    registry.register(_plugin("dup", version="1.0"))
    registry.register(_plugin("dup", version="2.0"))
    assert registry.registered_count == 1
    found = next(p for p in registry.discover() if p.name == "dup")
    assert found.version == "2.0"


def test_register_persists_all_fields(registry: PluginRegistry):
    registry.register(
        _plugin(
            "full",
            version="9.9",
            description="desc",
            author="bob",
            entry_point="mod:Cls",
            plugin_type="tool",
            enabled=False,
            config={"a": 1},
        )
    )
    found = next(p for p in registry.discover() if p.name == "full")
    assert found.version == "9.9"
    assert found.description == "desc"
    assert found.author == "bob"
    assert found.entry_point == "mod:Cls"
    assert found.plugin_type == "tool"
    assert found.enabled is False
    assert found.config == {"a": 1}
    assert found.installed is True


def test_register_none_config_stored_as_empty_dict(registry: PluginRegistry):
    registry.register(_plugin("nc", config=None))
    found = next(p for p in registry.discover() if p.name == "nc")
    assert found.config == {}


# ── unregister ─────────────────────────────────────────────────────


def test_unregister_existing_returns_true(registry: PluginRegistry):
    registry.register(_plugin("toremove"))
    assert registry.unregister("toremove") is True
    assert not registry.is_registered("toremove")
    assert registry.registered_count == 0


def test_unregister_missing_returns_false(registry: PluginRegistry):
    assert registry.unregister("ghost") is False


# ── enable / disable ───────────────────────────────────────────────


def test_disable_then_enable(registry: PluginRegistry):
    registry.register(_plugin("toggle"))
    assert registry.disable("toggle") is True
    found = next(p for p in registry.discover() if p.name == "toggle")
    assert found.enabled is False

    assert registry.enable("toggle") is True
    found = next(p for p in registry.discover() if p.name == "toggle")
    assert found.enabled is True


def test_enable_missing_returns_false(registry: PluginRegistry):
    assert registry.enable("nope") is False


def test_disable_missing_returns_false(registry: PluginRegistry):
    assert registry.disable("nope") is False


# ── discover ───────────────────────────────────────────────────────


def test_discover_includes_local(registry: PluginRegistry):
    with _no_entry_points():
        registry.register(_plugin("local1"))
        registry.register(_plugin("local2"))
        names = [p.name for p in registry.discover()]
    assert "local1" in names
    assert "local2" in names


def test_discover_empty_when_no_eps_and_no_local(registry: PluginRegistry):
    with _no_entry_points():
        assert registry.discover() == []


def test_discover_entry_points_select_path(registry: PluginRegistry):
    class _EP:
        name = "ep_cmd"
        dist = types.SimpleNamespace(version="3.0.0")

        def __str__(self):
            return "ep_module:Entry"

    def select(group):
        return [_EP()] if group == "llmstack.commands" else []

    fake = types.SimpleNamespace(select=select)
    with patch("importlib.metadata.entry_points", return_value=fake):
        plugins = registry.discover()

    ep_plugins = [p for p in plugins if p.name == "ep_cmd"]
    assert len(ep_plugins) == 1
    found = ep_plugins[0]
    assert found.version == "3.0.0"
    assert found.plugin_type == "command"
    assert found.installed is True
    assert found.entry_point == "ep_module:Entry"
    assert found.description == "Plugin: ep_cmd"


def test_discover_entry_points_no_dist_defaults_version(registry: PluginRegistry):
    class _EP:
        name = "nodist"
        dist = None

        def __str__(self):
            return "x:Y"

    def select(group):
        return [_EP()] if group == "llmstack.tools" else []

    fake = types.SimpleNamespace(select=select)
    with patch("importlib.metadata.entry_points", return_value=fake):
        plugins = registry.discover()
    found = next(p for p in plugins if p.name == "nodist")
    assert found.version == "0.0.0"
    assert found.plugin_type == "tool"


def test_discover_legacy_get_path(registry: PluginRegistry):
    """Container without .select uses the dict-style .get fallback."""

    class _EP:
        name = "legacy"
        dist = types.SimpleNamespace(version="0.5")

        def __str__(self):
            return "legacy_mod:Plug"

    class _Legacy(dict):
        # No `select` attribute -> code path falls through to .get().
        pass

    container = _Legacy()
    container["llmstack.providers"] = [_EP()]

    with patch("importlib.metadata.entry_points", return_value=container):
        plugins = registry.discover()
    legacy = [p for p in plugins if p.name == "legacy"]
    assert len(legacy) == 1
    assert legacy[0].plugin_type == "provider"


def test_discover_swallows_entry_point_errors(registry: PluginRegistry):
    with patch("importlib.metadata.entry_points", side_effect=RuntimeError("boom")):
        # Local plugins still returned despite EP discovery exploding.
        registry.register(_plugin("survivor"))
        plugins = registry.discover()
    assert [p.name for p in plugins] == ["survivor"]


# ── get_enabled ────────────────────────────────────────────────────


def test_get_enabled_filters_disabled(registry: PluginRegistry):
    with _no_entry_points():
        registry.register(_plugin("on", enabled=True))
        registry.register(_plugin("off", enabled=False))
        enabled = registry.get_enabled()
    names = [p.name for p in enabled]
    assert "on" in names
    assert "off" not in names


def test_get_enabled_filters_by_type(registry: PluginRegistry):
    with _no_entry_points():
        registry.register(_plugin("cmd1", plugin_type="command"))
        registry.register(_plugin("tool1", plugin_type="tool"))
        commands = registry.get_enabled(plugin_type="command")
    names = [p.name for p in commands]
    assert names == ["cmd1"]


# ── load_plugin ────────────────────────────────────────────────────


def test_load_plugin_module_attr(registry: PluginRegistry):
    sentinel = object()
    fake_module = types.SimpleNamespace(MyAttr=sentinel)
    with _no_entry_points():
        registry.register(_plugin("withattr", entry_point="some.mod:MyAttr"))
        with patch("importlib.import_module", return_value=fake_module) as imp:
            result = registry.load_plugin("withattr")
    assert result is sentinel
    imp.assert_called_once_with("some.mod")


def test_load_plugin_module_only(registry: PluginRegistry):
    fake_module = types.SimpleNamespace(value=42)
    with _no_entry_points():
        registry.register(_plugin("modonly", entry_point="just.a.module"))
        with patch("importlib.import_module", return_value=fake_module) as imp:
            result = registry.load_plugin("modonly")
    assert result is fake_module
    imp.assert_called_once_with("just.a.module")


def test_load_plugin_disabled_returns_none(registry: PluginRegistry):
    with _no_entry_points():
        registry.register(_plugin("disabled", enabled=False))
        result = registry.load_plugin("disabled")
    assert result is None


def test_load_plugin_unknown_returns_none(registry: PluginRegistry):
    with _no_entry_points():
        result = registry.load_plugin("nonexistent")
    assert result is None


def test_load_plugin_import_error_returns_none(registry: PluginRegistry):
    with _no_entry_points():
        registry.register(_plugin("broken", entry_point="bad.mod:Thing"))
        with patch("importlib.import_module", side_effect=ImportError("nope")):
            result = registry.load_plugin("broken")
    assert result is None


def test_load_plugin_missing_attr_returns_none(registry: PluginRegistry):
    fake_module = types.SimpleNamespace()  # no `Missing` attr
    with _no_entry_points():
        registry.register(_plugin("noattr", entry_point="m:Missing"))
        with patch("importlib.import_module", return_value=fake_module):
            result = registry.load_plugin("noattr")
    assert result is None


# ── round-trip config serialization ────────────────────────────────


def test_config_roundtrip_complex(registry: PluginRegistry):
    cfg = {"nested": {"list": [1, 2, 3]}, "flag": True, "s": "x"}
    with _no_entry_points():
        registry.register(_plugin("cfg", config=cfg))
        found = next(p for p in registry.discover() if p.name == "cfg")
    assert found.config == cfg
    # Confirm it was JSON-serialized in storage.
    with sqlite3.connect(registry.db_path) as conn:
        raw = conn.execute("SELECT config FROM plugins WHERE name = ?", ("cfg",)).fetchone()[0]
    assert json.loads(raw) == cfg
