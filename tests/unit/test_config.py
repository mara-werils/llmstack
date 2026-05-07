"""Tests for config schema and loader."""

import tempfile
from pathlib import Path

import yaml
import pytest

from llmstack.config.schema import StackConfig, ModelSpec, ModelsConfig
from llmstack.config.loader import save_config, load_config
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
