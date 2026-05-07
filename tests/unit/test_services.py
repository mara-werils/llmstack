"""Tests for service container specs and registry."""

from llmstack.config.schema import ModelSpec, EmbeddingSpec, VectorDBConfig, CacheConfig, GatewayConfig, ObserveConfig
from llmstack.core.hardware import HardwareProfile
from llmstack.services.inference.ollama import OllamaService
from llmstack.services.inference.vllm import VllmService
from llmstack.services.embeddings.tei import TEIService
from llmstack.services.vectordb.qdrant import QdrantService
from llmstack.services.cache.redis import RedisService
from llmstack.services.gateway.service import GatewayService
from llmstack.services.observe.prometheus import PrometheusService, GrafanaService
from llmstack.services.registry import ServiceRegistry


def _cpu_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor="none", gpu_name=None, gpu_vram_mb=0,
        cpu_cores=8, ram_mb=16384, os="linux", docker_runtime="default",
    )


def _nvidia_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor="nvidia", gpu_name="RTX 4090", gpu_vram_mb=24576,
        cpu_cores=16, ram_mb=65536, os="linux", docker_runtime="nvidia",
    )


# ── Ollama ──────────────────────────────────────────────────────

def test_ollama_container_spec():
    svc = OllamaService(ModelSpec(name="llama3.2"), _cpu_hw())
    spec = svc.container_spec()
    assert spec["image"] == "ollama/ollama:latest"
    assert "11434/tcp" in spec["ports"]
    assert "device_requests" not in spec  # no GPU


def test_ollama_openai_url():
    svc = OllamaService(ModelSpec(name="llama3.2"), _cpu_hw())
    assert "/v1" in svc.openai_base_url()


def test_ollama_health_url():
    svc = OllamaService(ModelSpec(name="llama3.2"), _cpu_hw())
    assert "11434" in svc.health_url()


# ── vLLM ────────────────────────────────────────────────────────

def test_vllm_container_spec():
    svc = VllmService(ModelSpec(name="meta-llama/Llama-3-8B"), _nvidia_hw())
    spec = svc.container_spec()
    assert spec["image"] == "vllm/vllm-openai:latest"
    assert "device_requests" in spec
    assert spec["shm_size"] == "4g"


def test_vllm_openai_url():
    svc = VllmService(ModelSpec(name="llama3"), _nvidia_hw())
    url = svc.openai_base_url()
    assert "/v1" in url


# ── TEI ─────────────────────────────────────────────────────────

def test_tei_container_spec_cpu():
    svc = TEIService(EmbeddingSpec(name="bge-m3"), _cpu_hw())
    spec = svc.container_spec()
    assert "cpu" in spec["image"]
    assert "device_requests" not in spec


def test_tei_container_spec_gpu():
    svc = TEIService(EmbeddingSpec(name="bge-m3"), _nvidia_hw())
    spec = svc.container_spec()
    assert "cpu" not in spec["image"]
    assert "device_requests" in spec


# ── Qdrant ──────────────────────────────────────────────────────

def test_qdrant_spec():
    svc = QdrantService(VectorDBConfig(port=6333))
    spec = svc.container_spec()
    assert spec["image"] == "qdrant/qdrant:latest"
    assert "6333/tcp" in spec["ports"]


def test_qdrant_health():
    svc = QdrantService(VectorDBConfig(port=6333))
    assert "healthz" in svc.health_url()


# ── Redis ───────────────────────────────────────────────────────

def test_redis_spec():
    svc = RedisService(CacheConfig(max_memory="512mb"))
    spec = svc.container_spec()
    assert spec["image"] == "redis:7-alpine"
    assert "512mb" in spec["command"]


# ── Gateway ─────────────────────────────────────────────────────

def test_gateway_spec():
    svc = GatewayService(
        config=GatewayConfig(port=8000, api_keys=["sk-test"]),
        inference_url="http://ollama:11434/v1",
        embeddings_url="http://ollama:11434/v1",
    )
    spec = svc.container_spec()
    assert spec["ports"]["8000/tcp"] == 8000
    assert "sk-test" in spec["environment"]["LLMSTACK_API_KEYS"]


def test_gateway_health():
    svc = GatewayService(
        config=GatewayConfig(port=9000),
        inference_url="", embeddings_url="",
    )
    assert "9000" in svc.health_url()


# ── Observability ───────────────────────────────────────────────

def test_prometheus_spec():
    svc = PrometheusService(ObserveConfig())
    spec = svc.container_spec()
    assert "prometheus" in spec["image"]
    assert "9090/tcp" in spec["ports"]


def test_grafana_spec():
    svc = GrafanaService(ObserveConfig(dashboard_port=3000))
    spec = svc.container_spec()
    assert "grafana" in spec["image"]
    assert spec["ports"]["3000/tcp"] == 3000


def test_prometheus_config_yaml():
    svc = PrometheusService(ObserveConfig())
    config = svc.get_config_yaml()
    assert "llmstack-gateway" in config
    assert "scrape_interval" in config


def test_grafana_dashboard_json():
    svc = GrafanaService(ObserveConfig())
    dashboard = svc.get_dashboard_json()
    assert "LLMStack" in dashboard
    assert "panels" in dashboard


# ── Registry ────────────────────────────────────────────────────

def test_registry_discovers_builtins():
    reg = ServiceRegistry()
    assert "ollama" in reg.all_names()
    assert "vllm" in reg.all_names()
    assert "qdrant" in reg.all_names()
    assert "redis" in reg.all_names()
    assert "tei" in reg.all_names()


def test_registry_get():
    reg = ServiceRegistry()
    cls = reg.get("ollama")
    assert cls is OllamaService


def test_registry_unknown_raises():
    reg = ServiceRegistry()
    try:
        reg.get("nonexistent")
        assert False, "Should have raised"
    except KeyError:
        pass


def test_registry_list_by_category():
    reg = ServiceRegistry()
    inference = reg.list_by_category("inference")
    names = [s.name for s in inference]
    assert "ollama" in names
    assert "vllm" in names
