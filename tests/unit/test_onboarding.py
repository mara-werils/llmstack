"""Tests for hardware-aware onboarding helpers."""

from __future__ import annotations

import httpx
import pytest

from llmstack.core.hardware import HardwareProfile
from llmstack.core.onboarding import (
    FIRST_VALUE_PROMPT,
    OllamaStatus,
    assess_readiness,
    chat_model_catalog,
    embed_model_catalog,
    ollama_install_hint,
    probe_ollama,
    recommend_embed_model,
    recommend_model,
    usable_memory_gb,
    verify_first_value,
)


def _hw(
    *,
    ram_gb: float = 16.0,
    gpu_vendor: str = "none",
    gpu_vram_gb: float = 0.0,
    cpu_cores: int = 8,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor=gpu_vendor,  # type: ignore[arg-type]
        gpu_name=None if gpu_vendor == "none" else gpu_vendor,
        gpu_vram_mb=int(gpu_vram_gb * 1024),
        cpu_cores=cpu_cores,
        ram_mb=int(ram_gb * 1024),
        os="linux",
        docker_runtime="default",
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- usable_memory_gb -------------------------------------------------------


def test_usable_memory_uses_vram_for_discrete_gpu():
    hw = _hw(ram_gb=64, gpu_vendor="nvidia", gpu_vram_gb=24)
    assert usable_memory_gb(hw) == pytest.approx(24.0)


def test_usable_memory_uses_ram_for_apple_unified_memory():
    hw = _hw(ram_gb=32, gpu_vendor="apple", gpu_vram_gb=32)
    assert usable_memory_gb(hw) == pytest.approx(32.0)


def test_usable_memory_uses_ram_for_cpu_only():
    hw = _hw(ram_gb=8, gpu_vendor="none")
    assert usable_memory_gb(hw) == pytest.approx(8.0)


# --- recommend_model --------------------------------------------------------


@pytest.mark.parametrize(
    ("ram_gb", "expected"),
    [
        (4, "llama3.2:1b"),
        (8, "llama3.2"),
        (16, "qwen2.5-coder:7b"),
        (32, "qwen2.5-coder:14b"),
        (128, "qwen2.5-coder:14b"),
    ],
)
def test_recommend_model_scales_with_memory(ram_gb, expected):
    assert recommend_model(_hw(ram_gb=ram_gb)).name == expected


def test_recommend_model_driven_by_gpu_vram_not_ram():
    # Plenty of RAM but a tiny GPU -> bounded by VRAM.
    hw = _hw(ram_gb=128, gpu_vendor="nvidia", gpu_vram_gb=6)
    assert recommend_model(hw).name == "llama3.2:1b"


def test_recommend_model_returns_smallest_for_tiny_machine():
    choice = recommend_model(_hw(ram_gb=2))
    assert choice.name == "llama3.2:1b"
    assert choice.approx_download_gb > 0
    assert choice.reason


# --- catalogs ---------------------------------------------------------------


def test_chat_catalog_is_non_empty_and_sorted_by_memory():
    catalog = chat_model_catalog()
    assert len(catalog) >= 2
    mins = [m.min_memory_gb for m in catalog]
    assert mins == sorted(mins)
    assert len({m.name for m in catalog}) == len(catalog)  # unique names


def test_embed_catalog_is_non_empty_and_sorted_by_memory():
    catalog = embed_model_catalog()
    assert len(catalog) >= 1
    mins = [m.min_memory_gb for m in catalog]
    assert mins == sorted(mins)


# --- recommend_embed_model --------------------------------------------------


def test_recommend_embed_model_default_runs_anywhere():
    choice = recommend_embed_model(_hw(ram_gb=4))
    assert choice.name == "nomic-embed-text"
    assert choice.approx_download_gb > 0


def test_recommend_embed_model_upgrades_on_capable_hardware():
    assert recommend_embed_model(_hw(ram_gb=32)).name == "mxbai-embed-large"


def test_recommend_embed_model_uses_vram_budget():
    hw = _hw(ram_gb=128, gpu_vendor="nvidia", gpu_vram_gb=8)
    assert recommend_embed_model(hw).name == "nomic-embed-text"


# --- OllamaStatus.has_model -------------------------------------------------


def test_has_model_exact_and_tag_prefix():
    status = OllamaStatus(running=True, models=("llama3.2:latest", "nomic-embed-text"))
    assert status.has_model("llama3.2")
    assert status.has_model("llama3.2:latest")
    assert status.has_model("nomic-embed-text")
    assert not status.has_model("qwen2.5-coder")


# --- probe_ollama -----------------------------------------------------------


def test_probe_ollama_running_lists_models():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3.2:1b"}, {"name": ""}]})

    status = probe_ollama(client=_client(handler))
    assert status.running is True
    assert status.models == ("llama3.2:1b",)  # blank names dropped
    assert status.error is None


def test_probe_ollama_not_running_on_connect_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    status = probe_ollama(client=_client(handler))
    assert status.running is False
    assert status.models == ()
    assert "connection refused" in (status.error or "")


def test_probe_ollama_not_running_on_bad_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    status = probe_ollama(client=_client(handler))
    assert status.running is False


def test_probe_ollama_creates_and_closes_its_own_client():
    # No client injected -> exercises the self-owned client + close() path.
    # 127.0.0.1:1 is loopback-only (no external egress) and always refuses.
    status = probe_ollama("http://127.0.0.1:1", timeout=0.2)
    assert status.running is False


# --- verify_first_value -----------------------------------------------------


def test_verify_first_value_returns_reply():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"response": "  Hello! I am local and private.  "})

    result = verify_first_value("llama3.2:1b", client=_client(handler))
    assert result.ok is True
    assert result.model == "llama3.2:1b"
    assert result.reply == "Hello! I am local and private."
    assert captured["path"] == "/api/generate"


def test_verify_first_value_empty_reply_is_not_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "   "})

    result = verify_first_value("llama3.2:1b", client=_client(handler))
    assert result.ok is False


def test_verify_first_value_reports_errors_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    result = verify_first_value("llama3.2:1b", client=_client(handler))
    assert result.ok is False
    assert "too slow" in (result.error or "")


def test_verify_first_value_creates_and_closes_its_own_client():
    # No client injected -> self-owned client + close() path; loopback refusal.
    result = verify_first_value("llama3.2:1b", "http://127.0.0.1:1", timeout=0.2)
    assert result.ok is False


def test_first_value_prompt_mentions_local_and_private():
    assert "local" in FIRST_VALUE_PROMPT
    assert "privately" in FIRST_VALUE_PROMPT


# --- assess_readiness -------------------------------------------------------


def test_assess_readiness_all_present_is_ready():
    hw = _hw(ram_gb=8)  # recommends llama3.2 + nomic-embed-text
    status = OllamaStatus(running=True, models=("llama3.2:latest", "nomic-embed-text:latest"))
    report = assess_readiness(hw, status)
    assert report.ready is True
    assert report.chat_model_ready and report.embed_model_ready
    assert any("llmstack ask" in h for h in report.hints)


def test_assess_readiness_ollama_down_lists_install_hints():
    report = assess_readiness(_hw(ram_gb=8), OllamaStatus(running=False))
    assert report.ready is False
    assert any("quickstart" in h for h in report.hints)
    assert any("ollama" in h.lower() for h in report.hints)


def test_assess_readiness_missing_models_suggests_pull():
    status = OllamaStatus(running=True, models=())
    report = assess_readiness(_hw(ram_gb=8), status)
    assert report.ready is False
    assert any(h.startswith("ollama pull llama3.2") for h in report.hints)
    assert any(h.startswith("ollama pull nomic-embed-text") for h in report.hints)


def test_readiness_summary_ready_and_not_ready():
    status_ready = OllamaStatus(running=True, models=("llama3.2:latest", "nomic-embed-text:latest"))
    ready = assess_readiness(_hw(ram_gb=8), status_ready)
    assert "Ready for zero-key local inference" in ready.summary()

    not_ready = assess_readiness(_hw(ram_gb=8), OllamaStatus(running=False))
    assert not_ready.summary().startswith("Not ready")


def test_assess_readiness_honours_explicit_models():
    status = OllamaStatus(running=True, models=("mistral:latest", "bge-m3:latest"))
    report = assess_readiness(_hw(ram_gb=8), status, chat_model="mistral", embed_model="bge-m3")
    assert report.ready is True
    assert report.chat_model == "mistral"
    assert report.embed_model == "bge-m3"


# --- ollama_install_hint ----------------------------------------------------


def test_install_hint_macos_uses_brew():
    hints = ollama_install_hint("Darwin")
    assert any("brew install ollama" in line for line in hints)
    assert "ollama serve" in hints


def test_install_hint_linux_uses_curl_installer():
    hints = ollama_install_hint("Linux")
    assert any("ollama.com/install.sh" in line for line in hints)


def test_install_hint_other_os_links_download():
    hints = ollama_install_hint("Windows")
    assert any("ollama.com/download" in line for line in hints)
