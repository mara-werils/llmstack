"""Tests for config schema and loader."""

import os
from unittest.mock import patch

import yaml
import pytest

from llmstack.config.schema import StackConfig, ModelSpec, ModelsConfig
from llmstack.config.loader import save_config, load_config, _deep_merge, _apply_env_overrides
from llmstack.config.presets import PRESETS


def test_default_config():
    config = StackConfig()
    assert config.version == "1"
    assert config.models.chat.name == "llama3.2"
    assert config.models.chat.backend == "auto"
    assert config.gateway.port == 8000
    assert config.gateway.auth == "api_key"


def test_custom_model():
    config = StackConfig(
        models=ModelsConfig(
            chat=ModelSpec(name="mistral:7b", backend="ollama"),
        )
    )
    assert config.models.chat.name == "mistral:7b"
    assert config.models.chat.backend == "ollama"


def test_presets_exist():
    assert "chat" in PRESETS
    assert "rag" in PRESETS
    assert "agent" in PRESETS


def test_rag_preset():
    config = PRESETS["rag"]
    assert config.models.chat.context_length == 8192
    assert config.models.embeddings.name == "bge-m3"


def test_agent_preset():
    config = PRESETS["agent"]
    assert "70b" in config.models.chat.name
    assert config.gateway.request_timeout == 300


def test_save_and_load(tmp_path):
    config = StackConfig(
        models=ModelsConfig(
            chat=ModelSpec(name="phi3", backend="ollama"),
        )
    )
    save_config(config, tmp_path)

    loaded = load_config(tmp_path)
    assert loaded.models.chat.name == "phi3"
    assert loaded.models.chat.backend == "ollama"
    assert loaded.gateway.port == 8000


def test_load_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)


def test_config_to_yaml():
    config = StackConfig()
    data = config.model_dump(mode="json")
    yaml_str = yaml.dump(data)
    assert "llama3.2" in yaml_str
    assert "qdrant" in yaml_str


# ---------------------------------------------------------------------------
# Config loader: local overrides and env var support
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        assert _deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"models": {"chat": {"name": "llama3.2", "backend": "auto"}}}
        override = {"models": {"chat": {"name": "llama3.1"}}}
        result = _deep_merge(base, override)
        assert result["models"]["chat"]["name"] == "llama3.1"
        assert result["models"]["chat"]["backend"] == "auto"

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1


class TestEnvOverrides:
    def test_string_override(self):
        raw = {"models": {"chat": {"name": "llama3.2"}}}
        with patch.dict(os.environ, {"LLMSTACK_CFG_MODELS__CHAT__NAME": "mistral"}):
            result = _apply_env_overrides(raw)
        assert result["models"]["chat"]["name"] == "mistral"

    def test_int_coercion(self):
        raw = {"gateway": {"port": 8000}}
        with patch.dict(os.environ, {"LLMSTACK_CFG_GATEWAY__PORT": "9000"}):
            result = _apply_env_overrides(raw)
        assert result["gateway"]["port"] == 9000

    def test_bool_coercion(self):
        raw = {"observe": {"metrics": True}}
        with patch.dict(os.environ, {"LLMSTACK_CFG_OBSERVE__METRICS": "false"}):
            result = _apply_env_overrides(raw)
        assert result["observe"]["metrics"] is False

    def test_empty_suffix_is_ignored(self):
        # A bare prefix or empty segment must not inject a "" key.
        raw = {"models": {}}
        with patch.dict(os.environ, {"LLMSTACK_CFG_": "x", "LLMSTACK_CFG___Y": "z"}):
            result = _apply_env_overrides(raw)
        assert "" not in result
        assert result == {"models": {}}


class TestLocalOverride:
    def test_load_with_local_override(self, tmp_path):
        config_file = tmp_path / "llmstack.yaml"
        config_file.write_text(
            yaml.dump({"version": "1", "models": {"chat": {"name": "llama3.2"}}})
        )

        local_file = tmp_path / "llmstack.local.yaml"
        local_file.write_text(yaml.dump({"models": {"chat": {"name": "mistral"}}}))

        config = load_config(tmp_path)
        assert config.models.chat.name == "mistral"

    def test_empty_yaml_uses_defaults(self, tmp_path):
        config_file = tmp_path / "llmstack.yaml"
        config_file.write_text("")
        config = load_config(tmp_path)
        assert config.gateway.port == 8000
