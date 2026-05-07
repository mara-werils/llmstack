"""Resolve 'auto' config values into concrete choices based on hardware."""

from __future__ import annotations

from llmstack.config.schema import ModelSpec, EmbeddingSpec
from llmstack.core.hardware import HardwareProfile


def resolve_inference_backend(model: ModelSpec, hw: HardwareProfile) -> str:
    """Pick the best inference backend for the detected hardware."""
    if model.backend != "auto":
        return model.backend

    # vLLM requires NVIDIA GPU with sufficient VRAM
    if hw.gpu_vendor == "nvidia" and hw.gpu_vram_mb >= 16_000:
        return "vllm"

    # Everything else: Ollama (supports CPU, Apple Silicon, smaller GPUs)
    return "ollama"


def resolve_embedding_backend(spec: EmbeddingSpec, hw: HardwareProfile) -> str:
    """Pick the best embedding backend."""
    if spec.backend != "auto":
        return spec.backend

    # TEI (Text Embeddings Inference) works well with GPU
    if hw.gpu_vendor == "nvidia" and hw.gpu_vram_mb >= 4_000:
        return "tei"

    # Fallback: use Ollama for embeddings too (simpler, works everywhere)
    return "ollama"


def resolve_quantization(model: ModelSpec, hw: HardwareProfile) -> str | None:
    """Auto-pick quantization based on available memory."""
    if model.quantization is not None:
        return model.quantization

    # Only auto-quantize for very large models on limited hardware
    model_lower = model.name.lower()
    if "70b" in model_lower:
        if hw.gpu_vram_mb < 48_000:
            return "q4_k_m"
    if "13b" in model_lower or "14b" in model_lower:
        if hw.gpu_vram_mb < 16_000:
            return "q4_k_m"

    return None
