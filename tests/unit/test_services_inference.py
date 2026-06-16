"""Coverage for inference services, registry, and ServiceBase helpers.

Targets the uncovered branches not exercised by test_services.py:
  - ollama.py: GPU passthrough + post_start model pull
  - vllm.py: quantization command branch + health_url
  - registry.py: entry-point plugin discovery (TypeError fallback + loop)
  - base.py: openai_base_url default + internal_url branches
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import llmstack.core  # noqa: F401  (import-order quirk: load core first)

from llmstack.config.schema import ModelSpec
from llmstack.core.hardware import HardwareProfile
from llmstack.services.base import ServiceBase
from llmstack.services.inference.ollama import OllamaService
from llmstack.services.inference.vllm import VllmService
from llmstack.services.registry import ServiceRegistry


def _cpu_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor="none",
        gpu_name=None,
        gpu_vram_mb=0,
        cpu_cores=8,
        ram_mb=16384,
        os="linux",
        docker_runtime="default",
    )


def _nvidia_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor="nvidia",
        gpu_name="RTX 4090",
        gpu_vram_mb=24576,
        cpu_cores=16,
        ram_mb=65536,
        os="linux",
        docker_runtime="nvidia",
    )


# ── Ollama: GPU passthrough (lines 36-38) ───────────────────────


def test_ollama_container_spec_gpu_adds_device_requests():
    svc = OllamaService(ModelSpec(name="llama3.2"), _nvidia_hw())
    spec = svc.container_spec()
    assert "device_requests" in spec
    assert len(spec["device_requests"]) == 1


# ── Ollama: post_start model pull (lines 47-56) ─────────────────


async def test_ollama_post_start_pulls_plain_model():
    svc = OllamaService(ModelSpec(name="llama3.2"), _cpu_hw())

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=client) as mk:
        await svc.post_start()

    mk.assert_called_once()
    url, kwargs = client.post.call_args[0][0], client.post.call_args[1]
    assert url.endswith("/api/pull")
    assert kwargs["json"]["name"] == "llama3.2"
    assert kwargs["json"]["stream"] is False
    resp.raise_for_status.assert_called_once()


async def test_ollama_post_start_pulls_quantized_model():
    svc = OllamaService(
        ModelSpec(name="llama3.2", quantization="q4_k_m"), _cpu_hw()
    )

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=client):
        await svc.post_start()

    assert client.post.call_args[1]["json"]["name"] == "llama3.2:q4_k_m"


async def test_ollama_post_start_raises_on_http_error():
    svc = OllamaService(ModelSpec(name="llama3.2"), _cpu_hw())

    resp = MagicMock()
    resp.raise_for_status = MagicMock(side_effect=RuntimeError("boom"))
    client = AsyncMock()
    client.post.return_value = resp
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=client):
        try:
            await svc.post_start()
            assert False, "should have raised"
        except RuntimeError as e:
            assert "boom" in str(e)


# ── vLLM: quantization branch (line 36) + health_url (line 56) ──


def test_vllm_container_spec_with_quantization():
    svc = VllmService(
        ModelSpec(name="meta-llama/Llama-3-8B", quantization="awq"),
        _nvidia_hw(),
    )
    spec = svc.container_spec()
    cmd = spec["command"]
    assert "--quantization" in cmd
    assert cmd[cmd.index("--quantization") + 1] == "awq"


def test_vllm_container_spec_without_quantization_omits_flag():
    svc = VllmService(ModelSpec(name="meta-llama/Llama-3-8B"), _nvidia_hw())
    assert "--quantization" not in svc.container_spec()["command"]


def test_vllm_health_url():
    svc = VllmService(ModelSpec(name="llama3"), _nvidia_hw())
    url = svc.health_url()
    assert "8001" in url
    assert url.endswith("/health")


# ── Registry: plugin entry-point discovery (lines 31-41) ────────


def _make_ep(name_attr: bool = True, load_raises: bool = False):
    ep = MagicMock()
    if load_raises:
        ep.load.side_effect = RuntimeError("bad plugin")
    else:
        cls = MagicMock()
        if name_attr:
            cls.name = "plugin-svc"
        else:
            del cls.name  # hasattr(cls, "name") -> False
        ep.load.return_value = cls
    return ep


def test_registry_loads_plugin_entry_point():
    ep = _make_ep()
    with patch(
        "llmstack.services.registry.entry_points", return_value=[ep]
    ):
        reg = ServiceRegistry()
    assert "plugin-svc" in reg.all_names()


def test_registry_skips_plugin_without_name_attr():
    ep = _make_ep(name_attr=False)
    with patch(
        "llmstack.services.registry.entry_points", return_value=[ep]
    ):
        reg = ServiceRegistry()
    # builtins still present, plugin (no .name) silently skipped
    assert "ollama" in reg.all_names()


def test_registry_swallows_plugin_load_errors():
    ep = _make_ep(load_raises=True)
    with patch(
        "llmstack.services.registry.entry_points", return_value=[ep]
    ):
        reg = ServiceRegistry()  # must not raise
    assert "ollama" in reg.all_names()


def test_registry_python311_typeerror_fallback():
    """entry_points(group=...) raising TypeError -> .get() fallback path."""
    ep = _make_ep()
    legacy = MagicMock()
    legacy.get.return_value = [ep]

    def fake_entry_points(*args, **kwargs):
        if "group" in kwargs:
            raise TypeError("group kwarg unsupported")
        return legacy

    with patch(
        "llmstack.services.registry.entry_points",
        side_effect=fake_entry_points,
    ):
        reg = ServiceRegistry()

    legacy.get.assert_called_once_with("llmstack.services", [])
    assert "plugin-svc" in reg.all_names()


# ── ServiceBase: openai_base_url default + internal_url branches ─


class _DefaultService(ServiceBase):
    """Uses ServiceBase.openai_base_url (returns None) + ports dict."""

    name = "default-svc"
    category = "misc"

    def container_spec(self) -> dict[str, Any]:
        return {"image": "x", "ports": {"8080/tcp": 8080}, "environment": {}}

    def health_url(self) -> str:
        return "http://localhost:8080"


class _NoPortsService(_DefaultService):
    name = "noports-svc"

    def container_spec(self) -> dict[str, Any]:
        return {"image": "x", "ports": {}, "environment": {}}


class _NonDictPortsService(_DefaultService):
    name = "listports-svc"

    def container_spec(self) -> dict[str, Any]:
        return {"image": "x", "ports": ["8080/tcp"], "environment": {}}


def test_base_openai_base_url_default_is_none():
    assert _DefaultService().openai_base_url() is None


def test_base_internal_url_with_dict_ports():
    assert _DefaultService().internal_url() == "http://default-svc:8080"


def test_base_internal_url_without_ports():
    assert _NoPortsService().internal_url() == "http://noports-svc"


def test_base_internal_url_non_dict_ports():
    # ports present but not a dict -> container_port stays None -> bare URL
    assert _NonDictPortsService().internal_url() == "http://listports-svc"


async def test_base_post_start_default_is_noop():
    # default ServiceBase.post_start is a no-op coroutine
    assert await _DefaultService().post_start() is None
