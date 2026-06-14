"""Tests for `llmstack init` — hardware-aware model recommendations and the wizard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from llmstack.cli.commands.init import init, recommend_models


def _hw(ram_gb: int, vram_mb: int = 0) -> SimpleNamespace:
    return SimpleNamespace(ram_mb=ram_gb * 1024, gpu_vram_mb=vram_mb, gpu_vendor="none")


def test_recommend_models_low_ram_offers_only_tiny_model() -> None:
    models = recommend_models(_hw(4))
    assert [m[0] for m in models] == ["llama3.2:1b"]


def test_recommend_models_midrange_recommends_8b_first() -> None:
    models = recommend_models(_hw(16))
    assert models[0][0] == "llama3.2"  # first == recommended default
    assert "llama3.2:1b" in {m[0] for m in models}  # fallback always present


def test_recommend_models_highend_offers_70b_first() -> None:
    models = recommend_models(_hw(64))
    assert models[0][0] == "llama3.1:70b"


def test_recommend_models_has_no_duplicates() -> None:
    names = [m[0] for m in recommend_models(_hw(128, vram_mb=80_000))]
    assert len(names) == len(set(names))


def test_init_yes_writes_default_config(tmp_path) -> None:
    init(preset=None, directory=tmp_path, yes=True)
    assert (tmp_path / "llmstack.yaml").exists()


def test_init_unknown_preset_exits(tmp_path) -> None:
    with pytest.raises(typer.Exit):
        init(preset="does-not-exist", directory=tmp_path, yes=True)


def test_wizard_applies_choices(monkeypatch) -> None:
    from rich.prompt import Confirm, IntPrompt

    # Pick option 1 for both prompts; enable privacy.
    monkeypatch.setattr(IntPrompt, "ask", classmethod(lambda cls, *a, **k: 1))
    monkeypatch.setattr(Confirm, "ask", classmethod(lambda cls, *a, **k: True))

    from llmstack.cli.commands.init import _run_wizard

    config = _run_wizard(_hw(16))
    assert config.models.chat.name == "llama3.2"  # first recommended model
    assert config.gateway.guardrails.enabled is True
