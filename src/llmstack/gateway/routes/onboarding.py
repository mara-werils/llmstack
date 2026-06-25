"""GET /v1/onboarding -- first-run readiness for zero-key local inference.

Lets any client (editor extension, CI, the web UI) ask "is this machine ready to
run locally, and if not, what's left?" using the same logic the CLI's quickstart
and doctor use, so every surface agrees.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from llmstack.core.hardware import detect_hardware
from llmstack.core.onboarding import (
    DEFAULT_OLLAMA_URL,
    assess_readiness,
    probe_ollama,
    recommend_embed_model,
    recommend_model,
)

router = APIRouter(tags=["Onboarding"])


@router.get("/onboarding")
async def onboarding(ollama_url: str | None = None):
    """Report readiness, recommended models, and concrete next steps."""
    url = ollama_url or os.getenv("LLMSTACK_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    hw = detect_hardware()
    status = probe_ollama(url)
    report = assess_readiness(hw, status)
    rec_chat = recommend_model(hw)
    rec_embed = recommend_embed_model(hw)

    return {
        "ready": report.ready,
        "ollama": {
            "url": url,
            "running": status.running,
            "models": list(status.models),
        },
        "hardware": {
            "cpu_cores": hw.cpu_cores,
            "ram_gb": round(hw.ram_gb, 1),
            "gpu_vendor": hw.gpu_vendor,
            "gpu_vram_gb": round(hw.gpu_vram_gb, 1),
        },
        "recommended": {
            "chat_model": {
                "name": rec_chat.name,
                "label": rec_chat.label,
                "reason": rec_chat.reason,
                "approx_download_gb": rec_chat.approx_download_gb,
            },
            "embed_model": {
                "name": rec_embed.name,
                "label": rec_embed.label,
                "reason": rec_embed.reason,
                "approx_download_gb": rec_embed.approx_download_gb,
            },
        },
        "chat_model": {"name": report.chat_model, "ready": report.chat_model_ready},
        "embed_model": {"name": report.embed_model, "ready": report.embed_model_ready},
        "hints": list(report.hints),
    }
