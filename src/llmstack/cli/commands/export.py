"""llmstack export — generate a standalone docker-compose.yml from llmstack.yaml."""

from __future__ import annotations

import yaml

from llmstack.cli.console import console
from llmstack.config.loader import load_config
from llmstack.config.schema import StackConfig
from llmstack.core.hardware import detect_hardware
from llmstack.core.resolver import resolve_inference_backend, resolve_embedding_backend


def _build_compose(config: StackConfig) -> dict:
    """Convert StackConfig into a docker-compose.yml dict."""
    hw = detect_hardware()
    inference_backend = resolve_inference_backend(config.models.chat, hw)
    embed_backend = resolve_embedding_backend(config.models.embeddings, hw)

    services: dict = {}
    volumes: dict = {}
    network_name = config.docker.network

    # 1. Qdrant
    services["qdrant"] = {
        "image": "qdrant/qdrant:latest",
        "container_name": "llmstack-qdrant",
        "ports": [f"{config.services.vectors.port}:6333", f"{config.services.vectors.port + 1}:6334"],
        "volumes": ["qdrant_data:/qdrant/storage"],
        "restart": "unless-stopped",
        "networks": [network_name],
    }
    volumes["qdrant_data"] = None

    # 2. Redis
    services["redis"] = {
        "image": "redis:7-alpine",
        "container_name": "llmstack-redis",
        "ports": [f"{config.services.cache.port}:6379"],
        "command": f"redis-server --maxmemory {config.services.cache.max_memory} --maxmemory-policy allkeys-lru",
        "restart": "unless-stopped",
        "networks": [network_name],
    }

    # 3. Inference
    if inference_backend == "vllm":
        model = config.models.chat
        cmd = f"--model {model.name} --host 0.0.0.0 --port 8000 --max-model-len {model.context_length}"
        if model.quantization:
            cmd += f" --quantization {model.quantization}"
        services["vllm"] = {
            "image": "vllm/vllm-openai:latest",
            "container_name": "llmstack-vllm",
            "ports": ["8001:8000"],
            "command": cmd,
            "volumes": ["vllm_cache:/root/.cache/huggingface"],
            "shm_size": "4g",
            "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}},
            "restart": "unless-stopped",
            "networks": [network_name],
        }
        volumes["vllm_cache"] = None
        inference_url = "http://vllm:8000/v1"
    else:
        svc: dict = {
            "image": "ollama/ollama:latest",
            "container_name": "llmstack-ollama",
            "ports": ["11434:11434"],
            "volumes": ["ollama_data:/root/.ollama"],
            "restart": "unless-stopped",
            "networks": [network_name],
        }
        if hw.gpu_vendor == "nvidia":
            svc["deploy"] = {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}}
        services["ollama"] = svc
        volumes["ollama_data"] = None
        inference_url = "http://ollama:11434/v1"

    # 4. Embeddings
    embeddings_url = inference_url  # fallback: use Ollama for embeddings
    if embed_backend == "tei":
        tei_image = "ghcr.io/huggingface/text-embeddings-inference:cpu-latest"
        tei_svc: dict = {
            "image": tei_image,
            "container_name": "llmstack-tei",
            "ports": ["8002:80"],
            "command": f"--model-id {config.models.embeddings.name} --port 80",
            "volumes": ["tei_cache:/data"],
            "restart": "unless-stopped",
            "networks": [network_name],
        }
        if hw.gpu_vendor == "nvidia":
            tei_svc["image"] = "ghcr.io/huggingface/text-embeddings-inference:latest"
            tei_svc["deploy"] = {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}}
        services["tei"] = tei_svc
        volumes["tei_cache"] = None
        embeddings_url = "http://tei:80/v1"

    # 5. Gateway
    qdrant_url = f"http://qdrant:{config.services.vectors.port}"
    redis_url = f"redis://redis:{config.services.cache.port}"
    api_keys = ",".join(config.gateway.api_keys) if config.gateway.api_keys else ""

    services["gateway"] = {
        "build": {
            "context": ".",
            "dockerfile": "src/llmstack/gateway/Dockerfile",
        },
        "container_name": "llmstack-gateway",
        "ports": [f"{config.gateway.port}:8000"],
        "environment": {
            "LLMSTACK_INFERENCE_URL": inference_url,
            "LLMSTACK_EMBEDDINGS_URL": embeddings_url,
            "LLMSTACK_QDRANT_URL": qdrant_url,
            "LLMSTACK_REDIS_URL": redis_url,
            "LLMSTACK_API_KEYS": api_keys,
            "LLMSTACK_CORS_ORIGINS": ",".join(config.gateway.cors),
            "LLMSTACK_REQUEST_TIMEOUT": str(config.gateway.request_timeout),
            "LLMSTACK_RATE_LIMIT": config.gateway.rate_limit,
        },
        "depends_on": ["qdrant", "redis"],
        "restart": "unless-stopped",
        "networks": [network_name],
    }
    # Add inference dependency
    if inference_backend == "vllm":
        services["gateway"]["depends_on"].append("vllm")
    else:
        services["gateway"]["depends_on"].append("ollama")
    if embed_backend == "tei":
        services["gateway"]["depends_on"].append("tei")

    # 6. Observability
    if config.observe.metrics:
        services["prometheus"] = {
            "image": "prom/prometheus:latest",
            "container_name": "llmstack-prometheus",
            "ports": ["9090:9090"],
            "command": [
                "--config.file=/etc/prometheus/prometheus.yml",
                f"--storage.tsdb.retention.time={config.observe.retention}",
                "--web.enable-lifecycle",
            ],
            "volumes": [
                "./prometheus.yml:/etc/prometheus/prometheus.yml:ro",
                "prometheus_data:/prometheus",
            ],
            "restart": "unless-stopped",
            "networks": [network_name],
        }
        volumes["prometheus_data"] = None

        services["grafana"] = {
            "image": "grafana/grafana:latest",
            "container_name": "llmstack-grafana",
            "ports": [f"{config.observe.dashboard_port}:3000"],
            "environment": {
                "GF_SECURITY_ADMIN_USER": "admin",
                "GF_SECURITY_ADMIN_PASSWORD": "llmstack",
                "GF_AUTH_ANONYMOUS_ENABLED": "true",
                "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
            },
            "restart": "unless-stopped",
            "networks": [network_name],
        }

    compose = {
        "version": "3.8",
        "services": services,
        "volumes": volumes,
        "networks": {network_name: {"driver": "bridge"}},
    }

    return compose


def export(output: str = "docker-compose.yml") -> None:
    """Generate a docker-compose.yml from the current llmstack.yaml."""
    config = load_config()
    compose = _build_compose(config)

    # Clean up None volume values for yaml output
    compose["volumes"] = {k: {} for k in compose["volumes"]}

    with open(output, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    service_count = len(compose["services"])
    console.print(f"\n[success]Exported {service_count} services to {output}[/]")
    console.print(f"Run with: [bold]docker compose -f {output} up -d[/]\n")
