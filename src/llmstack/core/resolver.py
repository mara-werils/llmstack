"""Resolve 'auto' config values into concrete choices based on hardware."""

from __future__ import annotations

import re

from llmstack.config.schema import ModelSpec, EmbeddingSpec
from llmstack.core.hardware import HardwareProfile


def _mentions_size(model_lower: str, *sizes: str) -> bool:
    """True if the model name mentions one of the parameter sizes as a whole token.

    Matches ``70b`` in ``llama3.1:70b`` or ``llama3.1:70b-instruct`` but not inside
    a larger run like ``70bit`` or ``coder-1170b``, so size lookalikes don't trigger
    auto-quantization. A size token must be bounded by a non-alphanumeric (or string
    edge) on both sides.
    """
    return any(re.search(rf"(?<![0-9a-z]){re.escape(s)}(?![0-9a-z])", model_lower) for s in sizes)


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
    if _mentions_size(model_lower, "70b"):
        if hw.gpu_vram_mb < 48_000:
            return "q4_k_m"
    if _mentions_size(model_lower, "13b", "14b"):
        if hw.gpu_vram_mb < 16_000:
            return "q4_k_m"
    if _mentions_size(model_lower, "7b", "8b"):
        if hw.gpu_vram_mb < 8_000:
            return "q4_k_m"

    return None
