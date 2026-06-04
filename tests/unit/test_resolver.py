"""Tests for backend resolver."""

from llmstack.config.schema import ModelSpec, EmbeddingSpec
from llmstack.core.hardware import HardwareProfile
from llmstack.core.resolver import (
    resolve_inference_backend,
    resolve_embedding_backend,
    resolve_quantization,
)


def _make_hw(**kwargs) -> HardwareProfile:
    defaults = dict(
        gpu_vendor="none",
        gpu_name=None,
        gpu_vram_mb=0,
        cpu_cores=8,
        ram_mb=16384,
        os="linux",
        docker_runtime="default",
    )
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


def test_auto_picks_ollama_for_cpu():
    hw = _make_hw(gpu_vendor="none")
    model = ModelSpec(name="llama3.2", backend="auto")
    assert resolve_inference_backend(model, hw) == "ollama"


def test_auto_picks_vllm_for_large_nvidia():
    hw = _make_hw(gpu_vendor="nvidia", gpu_name="RTX 4090", gpu_vram_mb=24576)
    model = ModelSpec(name="llama3.2", backend="auto")
    assert resolve_inference_backend(model, hw) == "vllm"


def test_auto_picks_ollama_for_small_nvidia():
    hw = _make_hw(gpu_vendor="nvidia", gpu_name="RTX 3060", gpu_vram_mb=12288)
    model = ModelSpec(name="llama3.2", backend="auto")
    assert resolve_inference_backend(model, hw) == "ollama"


def test_auto_picks_ollama_for_apple():
    hw = _make_hw(gpu_vendor="apple", gpu_name="Apple M2", gpu_vram_mb=16384)
    model = ModelSpec(name="llama3.2", backend="auto")
    assert resolve_inference_backend(model, hw) == "ollama"


def test_explicit_backend_honored():
    hw = _make_hw(gpu_vendor="none")
    model = ModelSpec(name="llama3.2", backend="vllm")
    assert resolve_inference_backend(model, hw) == "vllm"


def test_auto_quantizes_70b_on_small_gpu():
    hw = _make_hw(gpu_vendor="nvidia", gpu_vram_mb=24576)
    model = ModelSpec(name="llama3.1:70b", backend="auto")
    assert resolve_quantization(model, hw) == "q4_k_m"


def test_no_quantize_small_model():
    hw = _make_hw(gpu_vendor="nvidia", gpu_vram_mb=24576)
    model = ModelSpec(name="llama3.2", backend="auto")
    assert resolve_quantization(model, hw) is None


def test_embedding_tei_on_nvidia():
    hw = _make_hw(gpu_vendor="nvidia", gpu_vram_mb=8000)
    spec = EmbeddingSpec(name="bge-m3", backend="auto")
    assert resolve_embedding_backend(spec, hw) == "tei"


def test_embedding_ollama_on_cpu():
    hw = _make_hw(gpu_vendor="none")
    spec = EmbeddingSpec(name="bge-m3", backend="auto")
    assert resolve_embedding_backend(spec, hw) == "ollama"
