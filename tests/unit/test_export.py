"""Tests for docker-compose export."""

from llmstack.cli.commands.export import _build_compose
from llmstack.config.schema import StackConfig


def test_export_default_config():
    config = StackConfig()
    compose = _build_compose(config)
    assert "services" in compose
    assert "volumes" in compose
    assert "networks" in compose


def test_export_has_core_services():
    config = StackConfig()
    compose = _build_compose(config)
    services = compose["services"]
    assert "qdrant" in services
    assert "redis" in services
    assert "gateway" in services
    # Either ollama or vllm should be present
    assert "ollama" in services or "vllm" in services


def test_export_gateway_depends_on_inference():
    config = StackConfig()
    compose = _build_compose(config)
    deps = compose["services"]["gateway"]["depends_on"]
    assert "qdrant" in deps
    assert "redis" in deps
    assert "ollama" in deps or "vllm" in deps


def test_export_observability_when_enabled():
    config = StackConfig()
    config.observe.metrics = True
    compose = _build_compose(config)
    assert "prometheus" in compose["services"]
    assert "grafana" in compose["services"]


def test_export_no_observability_when_disabled():
    config = StackConfig()
    config.observe.metrics = False
    compose = _build_compose(config)
    assert "prometheus" not in compose["services"]
    assert "grafana" not in compose["services"]


def test_export_redis_command_has_maxmemory():
    config = StackConfig()
    config.services.cache.max_memory = "512mb"
    compose = _build_compose(config)
    assert "512mb" in compose["services"]["redis"]["command"]


def test_export_gateway_env_vars():
    config = StackConfig()
    compose = _build_compose(config)
    env = compose["services"]["gateway"]["environment"]
    assert "LLMSTACK_INFERENCE_URL" in env
    assert "LLMSTACK_EMBEDDINGS_URL" in env
    assert "LLMSTACK_QDRANT_URL" in env
    assert "LLMSTACK_REDIS_URL" in env
