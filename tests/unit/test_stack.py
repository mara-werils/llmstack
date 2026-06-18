"""Tests for llmstack.core.stack.Stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmstack.config.schema import StackConfig
from llmstack.core.stack import Stack


@pytest.fixture
def patched_stack():
    hw = MagicMock()
    with (
        patch("llmstack.core.stack.detect_hardware", return_value=hw),
        patch("llmstack.core.stack.DockerManager") as docker_cls,
    ):
        docker_mgr = MagicMock()
        docker_cls.return_value = docker_mgr
        stack = Stack(StackConfig())
        yield stack, docker_mgr, hw


def _service(name, category, openai_url=None, health_url="http://svc/health", build_info=None):
    svc = MagicMock()
    svc.name = name
    svc.category = category
    svc.openai_base_url.return_value = openai_url
    svc.health_url.return_value = health_url
    if build_info is None:
        del svc.build_info
    else:
        svc.build_info.return_value = build_info
    svc.post_start = AsyncMock()
    return svc


def test_init_builds_docker_manager_with_network(patched_stack):
    stack, docker_mgr, hw = patched_stack
    assert stack.docker is docker_mgr
    assert stack.hw is hw
    assert stack._services == []


def test_build_services_ollama_backend(patched_stack):
    stack, _, _ = patched_stack
    with (
        patch("llmstack.core.stack.resolve_inference_backend", return_value="ollama"),
        patch("llmstack.core.stack.resolve_embedding_backend", return_value="tei"),
    ):
        services = stack._build_services()

    categories = [s.category for s in services]
    assert categories == ["vectordb", "cache", "inference", "embeddings", "gateway", "observe", "observe"]


def test_build_services_vllm_backend(patched_stack):
    stack, _, _ = patched_stack
    with (
        patch("llmstack.core.stack.resolve_inference_backend", return_value="vllm"),
        patch("llmstack.core.stack.resolve_embedding_backend", return_value="ollama"),
    ):
        services = stack._build_services()

    # embed_backend == "ollama" -> no extra embeddings service
    categories = [s.category for s in services]
    assert "embeddings" not in categories
    from llmstack.services.inference.vllm import VllmService

    assert any(isinstance(s, VllmService) for s in services)


def test_build_services_no_observability(patched_stack):
    stack, _, _ = patched_stack
    stack.config.observe.metrics = False
    with (
        patch("llmstack.core.stack.resolve_inference_backend", return_value="ollama"),
        patch("llmstack.core.stack.resolve_embedding_backend", return_value="ollama"),
    ):
        services = stack._build_services()

    assert "observe" not in [s.category for s in services]


def test_resolve_inference_url_found(patched_stack):
    stack, _, _ = patched_stack
    services = [_service("ollama", "inference", openai_url="http://ollama:11434/v1")]
    assert stack._resolve_inference_url(services, backend="ollama") == "http://ollama:11434/v1"


def test_resolve_inference_url_not_found(patched_stack):
    stack, _, _ = patched_stack
    assert stack._resolve_inference_url([], backend="ollama") == ""


def test_resolve_embeddings_url_direct(patched_stack):
    stack, _, _ = patched_stack
    services = [_service("tei", "embeddings", openai_url="http://tei:8080/v1")]
    assert stack._resolve_embeddings_url(services, "tei") == "http://tei:8080/v1"


def test_resolve_embeddings_url_falls_back_to_inference(patched_stack):
    stack, _, _ = patched_stack
    services = [_service("ollama", "inference", openai_url="http://ollama:11434/v1")]
    assert stack._resolve_embeddings_url(services, "ollama") == "http://ollama:11434/v1"


def test_resolve_embeddings_url_none_found(patched_stack):
    stack, _, _ = patched_stack
    assert stack._resolve_embeddings_url([], "ollama") == ""


def test_get_inference_url(patched_stack):
    stack, _, _ = patched_stack
    stack._services = [_service("ollama", "inference", openai_url="http://ollama:11434/v1")]
    assert stack._get_inference_url() == "http://ollama:11434/v1"


def test_get_inference_url_none(patched_stack):
    stack, _, _ = patched_stack
    stack._services = []
    assert stack._get_inference_url() == ""


def test_get_embeddings_url_direct(patched_stack):
    stack, _, _ = patched_stack
    stack._services = [_service("tei", "embeddings", openai_url="http://tei:8080/v1")]
    assert stack._get_embeddings_url() == "http://tei:8080/v1"


def test_get_embeddings_url_falls_back_to_ollama_instance(patched_stack):
    from llmstack.services.inference.ollama import OllamaService

    stack, _, _ = patched_stack
    ollama_svc = MagicMock(spec=OllamaService)
    ollama_svc.category = "inference"
    ollama_svc.openai_base_url.return_value = "http://ollama:11434/v1"
    stack._services = [ollama_svc]
    assert stack._get_embeddings_url() == "http://ollama:11434/v1"


def test_get_embeddings_url_no_match(patched_stack):
    stack, _, _ = patched_stack
    stack._services = [_service("vllm", "inference", openai_url="http://vllm:8000/v1")]
    assert stack._get_embeddings_url() == ""


def test_down_delegates_to_docker(patched_stack):
    stack, docker_mgr, _ = patched_stack
    docker_mgr.stop_all.return_value = ["llmstack-ollama"]
    assert stack.down(remove_volumes=True) == ["llmstack-ollama"]
    docker_mgr.stop_all.assert_called_once_with(remove_volumes=True)


def test_status_parses_container_info(patched_stack):
    stack, docker_mgr, _ = patched_stack
    docker_mgr.list_services.return_value = [
        {
            "name": "ollama",
            "container_id": "abc123",
            "status": "running",
            "ports": {"11434/tcp": [{"HostPort": "11434"}]},
        },
        {
            "name": "redis",
            "container_id": "def456",
            "status": "exited",
            "ports": {},
        },
    ]

    statuses = stack.status()

    assert statuses[0].name == "ollama"
    assert statuses[0].port == 11434
    from llmstack.services.base import ServiceState

    assert statuses[0].state == ServiceState.RUNNING
    assert statuses[1].state == ServiceState.STOPPED
    assert statuses[1].port is None


@pytest.mark.asyncio
async def test_up_happy_path_generates_api_key(patched_stack):
    stack, docker_mgr, _ = patched_stack
    inference_svc = _service("ollama", "inference", openai_url="http://ollama:11434/v1")
    cache_svc = _service("redis", "cache")
    services = [cache_svc, inference_svc]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock(return_value=True)),
        patch("llmstack.core.stack.save_config") as mock_save,
    ):
        await stack.up()

    docker_mgr.ensure_network.assert_called_once()
    assert docker_mgr.run_service.call_count == 2
    assert stack.config.gateway.api_keys[0].startswith("sk-llmstack-")
    mock_save.assert_called_once_with(stack.config)
    inference_svc.post_start.assert_awaited_once()
    cache_svc.post_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_up_skips_api_key_gen_when_keys_present(patched_stack):
    stack, docker_mgr, _ = patched_stack
    stack.config.gateway.api_keys = ["sk-existing"]
    services = [_service("redis", "cache")]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock(return_value=True)),
        patch("llmstack.core.stack.save_config") as mock_save,
    ):
        await stack.up()

    mock_save.assert_not_called()
    assert stack.config.gateway.api_keys == ["sk-existing"]


@pytest.mark.asyncio
async def test_up_skips_api_key_gen_when_auth_none(patched_stack):
    stack, docker_mgr, _ = patched_stack
    stack.config.gateway.auth = "none"
    services = [_service("redis", "cache")]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock(return_value=True)),
        patch("llmstack.core.stack.save_config") as mock_save,
    ):
        await stack.up()

    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_up_skips_health_check_for_cache(patched_stack):
    stack, docker_mgr, _ = patched_stack
    cache_svc = _service("redis", "cache")
    services = [cache_svc]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock()) as mock_wait,
        patch("llmstack.core.stack.save_config"),
    ):
        await stack.up()

    mock_wait.assert_not_called()


@pytest.mark.asyncio
async def test_up_raises_when_health_check_fails(patched_stack):
    stack, docker_mgr, _ = patched_stack
    inference_svc = _service("ollama", "inference")
    services = [inference_svc]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock(return_value=False)),
        patch("llmstack.core.stack.save_config"),
    ):
        with pytest.raises(RuntimeError, match="failed to start"):
            await stack.up()


@pytest.mark.asyncio
async def test_up_builds_local_image_when_build_info_present(patched_stack):
    stack, docker_mgr, _ = patched_stack
    gw_svc = _service(
        "gateway",
        "gateway",
        build_info={"path": "/pkg", "dockerfile": "/pkg/Dockerfile", "tag": "img:local"},
    )
    services = [gw_svc]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock(return_value=True)),
        patch("llmstack.core.stack.save_config"),
    ):
        await stack.up()

    docker_mgr.build_image.assert_called_once_with(
        path="/pkg", dockerfile="/pkg/Dockerfile", tag="img:local"
    )


@pytest.mark.asyncio
async def test_up_pulls_model_for_inference_service(patched_stack):
    stack, docker_mgr, _ = patched_stack
    inference_svc = _service("ollama", "inference")
    services = [inference_svc]

    with (
        patch.object(stack, "_build_services", return_value=services),
        patch("llmstack.core.stack.wait_healthy", new=AsyncMock(return_value=True)),
        patch("llmstack.core.stack.save_config"),
    ):
        await stack.up()

    inference_svc.post_start.assert_awaited_once()


def test_print_summary_with_inference_service(patched_stack, capsys):
    stack, _, _ = patched_stack
    stack._services = [
        _service("ollama", "inference", health_url="http://ollama:11434/api/tags"),
        _service("redis", "cache", health_url="http://redis:6379/health"),
    ]

    stack._print_summary()

    out = capsys.readouterr().out
    assert "LLMStack Services" in out
    assert "Try it:" in out
    assert "/v1/chat/completions" in out


def test_print_summary_without_inference_service(patched_stack, capsys):
    stack, _, _ = patched_stack
    stack._services = [_service("redis", "cache")]

    stack._print_summary()

    out = capsys.readouterr().out
    assert "Try it:" not in out
