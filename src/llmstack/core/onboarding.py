"""Hardware-aware onboarding helpers.

The verified #1 differentiator for adoption is "one command to first value with
zero API keys." These helpers keep that flow honest and testable:

- pick a local model sized to the detected hardware (no guessing, no OOM),
- probe whether Ollama is reachable and which models are already pulled,
- prove first value by running a real local completion and returning its reply.

Everything here is dependency-light and accepts an injectable ``httpx`` client so
the ``quickstart`` command can be exercised without a running Ollama. None of this
needs Docker -- Docker is only required for the full gateway stack (``llmstack up``).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import httpx

from llmstack.core.hardware import HardwareProfile

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ModelChoice:
    """A recommended local model and why it was chosen."""

    name: str  # Ollama tag, e.g. "llama3.2:1b"
    label: str  # human-friendly name, e.g. "Llama 3.2 1B"
    min_memory_gb: float  # usable memory (VRAM or RAM) this model wants
    approx_download_gb: float
    reason: str


# Ordered small -> large by the memory each model wants. ``recommend_model`` picks
# the largest entry that fits the detected hardware. Tags are current, widely
# pulled Ollama models that are good for general + coding use.
_CATALOG: tuple[ModelChoice, ...] = (
    ModelChoice("llama3.2:1b", "Llama 3.2 1B", 0.0, 1.3, "runs anywhere, fast even on CPU"),
    ModelChoice("llama3.2", "Llama 3.2 3B", 8.0, 2.0, "balanced general-purpose default"),
    ModelChoice("qwen2.5-coder:7b", "Qwen2.5 Coder 7B", 16.0, 4.7, "strong coding model"),
    ModelChoice("qwen2.5-coder:14b", "Qwen2.5 Coder 14B", 32.0, 9.0, "best local coding quality"),
)


def usable_memory_gb(hw: HardwareProfile) -> float:
    """Memory a model can realistically use.

    A discrete GPU (NVIDIA/AMD) is bounded by its VRAM. Apple Silicon and CPU-only
    machines draw from system RAM (unified memory on Apple), so use RAM there.
    """
    if hw.gpu_vendor in ("nvidia", "amd") and hw.gpu_vram_mb > 0:
        return hw.gpu_vram_mb / 1024.0
    return hw.ram_mb / 1024.0


def recommend_model(hw: HardwareProfile) -> ModelChoice:
    """Pick the largest catalog model whose memory budget fits the hardware."""
    memory = usable_memory_gb(hw)
    choice = _CATALOG[0]
    for model in _CATALOG:
        if memory >= model.min_memory_gb:
            choice = model
    return choice


@dataclass(frozen=True)
class EmbedChoice:
    """A recommended local embedding model for ``ask``/RAG and why."""

    name: str  # Ollama tag, e.g. "nomic-embed-text"
    label: str
    min_memory_gb: float
    approx_download_gb: float
    reason: str


# ``ask`` and RAG need an embedding model. nomic-embed-text is tiny and runs
# anywhere; capable machines get a higher-quality retriever.
_EMBED_CATALOG: tuple[EmbedChoice, ...] = (
    EmbedChoice("nomic-embed-text", "Nomic Embed Text", 0.0, 0.27, "fast, runs anywhere"),
    EmbedChoice("mxbai-embed-large", "MxBai Embed Large", 16.0, 0.67, "higher-quality retrieval"),
)


def recommend_embed_model(hw: HardwareProfile) -> EmbedChoice:
    """Pick the largest embedding model whose memory budget fits the hardware."""
    memory = usable_memory_gb(hw)
    choice = _EMBED_CATALOG[0]
    for model in _EMBED_CATALOG:
        if memory >= model.min_memory_gb:
            choice = model
    return choice


@dataclass(frozen=True)
class OllamaStatus:
    """Result of probing a local Ollama server."""

    running: bool
    models: tuple[str, ...] = ()
    error: str | None = None

    def has_model(self, name: str) -> bool:
        """True if ``name`` is installed, matching the bare name or a ``name:tag``."""
        return any(m == name or m.startswith(f"{name}:") for m in self.models)


def probe_ollama(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    *,
    client: httpx.Client | None = None,
    timeout: float = 5.0,
) -> OllamaStatus:
    """Report whether Ollama is reachable and which models are pulled.

    Never raises: any connection or protocol error becomes ``running=False`` so the
    caller can show install guidance instead of a traceback.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        resp = client.get(f"{ollama_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = tuple(
            name for m in resp.json().get("models", []) if (name := m.get("name"))
        )
        return OllamaStatus(running=True, models=models)
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        return OllamaStatus(running=False, error=str(exc))
    finally:
        if owns_client:
            client.close()


FIRST_VALUE_PROMPT = (
    "Reply in one short sentence: say hello and confirm you are a local model "
    "running privately on this machine."
)


@dataclass(frozen=True)
class FirstValue:
    """Outcome of the first real local completion -- the "it works" moment."""

    ok: bool
    model: str
    reply: str = ""
    error: str | None = None


def verify_first_value(
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    *,
    prompt: str = FIRST_VALUE_PROMPT,
    client: httpx.Client | None = None,
    timeout: float = 120.0,
) -> FirstValue:
    """Run one real local completion and return its reply.

    Proves end-to-end that inference works with zero API keys and zero egress.
    Never raises: failures surface as ``ok=False`` with an ``error`` message.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        resp = client.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        reply = (resp.json().get("response") or "").strip()
        return FirstValue(ok=bool(reply), model=model, reply=reply)
    except Exception as exc:  # noqa: BLE001 - report instead of crashing onboarding
        return FirstValue(ok=False, model=model, error=str(exc))
    finally:
        if owns_client:
            client.close()


@dataclass(frozen=True)
class ReadinessReport:
    """Whether the machine is ready for zero-key local inference, and what's left.

    The single source of truth reused by ``doctor``, the gateway readiness route,
    the SDKs, the MCP tool, and the editor extensions so every surface agrees.
    """

    ollama_running: bool
    chat_model: str
    chat_model_ready: bool
    embed_model: str
    embed_model_ready: bool
    ready: bool
    hints: tuple[str, ...]


def assess_readiness(
    hw: HardwareProfile,
    status: OllamaStatus,
    *,
    chat_model: str | None = None,
    embed_model: str | None = None,
) -> ReadinessReport:
    """Combine hardware, Ollama status, and model availability into one verdict."""
    chat = chat_model or recommend_model(hw).name
    embed = embed_model or recommend_embed_model(hw).name
    chat_ready = status.has_model(chat)
    embed_ready = status.has_model(embed)
    ready = status.running and chat_ready and embed_ready

    hints: list[str] = []
    if not status.running:
        hints.extend(ollama_install_hint())
        hints.append("Then run: llmstack quickstart")
    else:
        if not chat_ready:
            hints.append(f"ollama pull {chat}")
        if not embed_ready:
            hints.append(f"ollama pull {embed}")
    if ready:
        hints.append("Ready -- try: llmstack ask -i .")

    return ReadinessReport(
        ollama_running=status.running,
        chat_model=chat,
        chat_model_ready=chat_ready,
        embed_model=embed,
        embed_model_ready=embed_ready,
        ready=ready,
        hints=tuple(hints),
    )


def ollama_install_hint(system: str | None = None) -> list[str]:
    """OS-specific commands to install and start Ollama."""
    system = system or platform.system()
    if system == "Darwin":
        return [
            "brew install ollama   # or download: https://ollama.com/download",
            "ollama serve",
        ]
    if system == "Linux":
        return [
            "curl -fsSL https://ollama.com/install.sh | sh",
            "ollama serve",
        ]
    return [
        "Download Ollama: https://ollama.com/download",
        "Start it, then re-run: llmstack quickstart",
    ]
