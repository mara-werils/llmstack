"""LLMStack Gateway — OpenAI-compatible API gateway with caching, RAG, and resilience."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmstack.gateway.routes.chat import router as chat_router
from llmstack.gateway.routes.embeddings import router as embeddings_router
from llmstack.gateway.routes.models import router as models_router
from llmstack.gateway.routes.health import router as health_router
from llmstack.gateway.routes.rag import router as rag_router
from llmstack.gateway.routes.router import router as router_router
from llmstack.gateway.routes.observe import router as observe_router
from llmstack.gateway.routes.learn import router as learn_router
from llmstack.gateway.routes.templates import router as templates_router
from llmstack.gateway.routes.conversations import router as conversations_router
from llmstack.gateway.routes.cost import router as cost_router
from llmstack.gateway.routes.webhooks import router as webhooks_router
from llmstack.gateway.routes.batch import router as batch_router
from llmstack.gateway.routes.leaderboard import router as leaderboard_router
from llmstack.gateway.routes.widget import router as widget_router
from llmstack.gateway.middleware.auth import AuthMiddleware
from llmstack.gateway.middleware.metrics import MetricsMiddleware
from llmstack.gateway.middleware.rate_limit import RateLimitMiddleware
from llmstack.gateway.middleware.logging import LoggingMiddleware
from llmstack.gateway.middleware.request_size import RequestSizeMiddleware
from llmstack.gateway.middleware.correlation import CorrelationMiddleware

logger = logging.getLogger(__name__)

_UI_DIR = Path(__file__).resolve().parent / "ui"


def _init_router() -> None:
    """Initialise the smart model router from env config (if provided).

    The router config is passed via the ``LLMSTACK_ROUTER_CONFIG`` env var
    as a JSON string, e.g.::

        {
            "enabled": true,
            "strategy": "cost",
            "models": [
                {"name": "llama3.2:1b", "tier": "simple", "max_context": 8192,
                 "speed_score": 3.0, "quality_score": 0.6},
                {"name": "llama3.2", "tier": "medium", "max_context": 8192,
                 "speed_score": 1.5, "quality_score": 0.85},
                {"name": "llama3.1:70b", "tier": "complex", "max_context": 16384,
                 "speed_score": 0.3, "quality_score": 1.0}
            ]
        }
    """
    raw = os.getenv("LLMSTACK_ROUTER_CONFIG", "")
    if not raw:
        return

    try:
        cfg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLMSTACK_ROUTER_CONFIG is set but not valid JSON — router disabled")
        return

    if not cfg.get("enabled", False):
        return

    from llmstack.gateway.router.router import ModelRouter, ModelTier
    from llmstack.gateway.router.stats import RouterStats
    from llmstack.gateway.router._state import init_router

    model_defs = cfg.get("models", [])
    if not model_defs:
        logger.warning("Router enabled but no models configured — router disabled")
        return

    tiers = [
        ModelTier(
            name=m["name"],
            tier=m.get("tier", "medium"),
            max_context=m.get("max_context", 8192),
            speed_score=m.get("speed_score", 1.0),
            quality_score=m.get("quality_score", 1.0),
            provider=m.get("provider", "local"),
            cost_per_m_input=m.get("cost_per_m_input", 0.0),
            cost_per_m_output=m.get("cost_per_m_output", 0.0),
        )
        for m in model_defs
    ]

    strategy = cfg.get("strategy", "cost")
    router = ModelRouter(models=tiers, strategy=strategy)
    stats = RouterStats()

    # Identify the largest model for savings tracking
    largest = max(tiers, key=lambda t: t.quality_score)
    stats.set_largest_model(largest.name)

    init_router(router, stats)
    logger.info(
        "Smart Model Router initialised: %d models, strategy=%s",
        len(tiers), strategy,
    )


def _init_providers() -> None:
    """Initialise the provider registry from env config (if provided).

    The provider config is passed via ``LLMSTACK_PROVIDERS_CONFIG`` env var
    as a JSON string, e.g.::

        {
            "enabled": true,
            "providers": [
                {"name": "openai", "api_key": "sk-...", "models": [...]},
                {"name": "anthropic", "api_key": "sk-ant-...", "models": [...]}
            ]
        }
    """
    raw = os.getenv("LLMSTACK_PROVIDERS_CONFIG", "")
    if not raw:
        return

    try:
        cfg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLMSTACK_PROVIDERS_CONFIG is set but not valid JSON — providers disabled")
        return

    if not cfg.get("enabled", False):
        return

    from llmstack.gateway.providers.registry import ProviderRegistry, init_registry
    from llmstack.gateway.providers.local import LocalProvider
    from llmstack.gateway.providers.openai_provider import OpenAIProvider
    from llmstack.gateway.providers.anthropic_provider import AnthropicProvider
    from llmstack.gateway.providers.google_provider import GoogleProvider
    from llmstack.gateway.providers.openai_compat import (
        GroqProvider, TogetherProvider, MistralProvider,
    )

    _PROVIDER_CLASSES = {
        "local": LocalProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "groq": GroqProvider,
        "together": TogetherProvider,
        "mistral": MistralProvider,
    }

    registry = ProviderRegistry()

    # Always register local provider
    registry.register(LocalProvider())

    provider_defs = cfg.get("providers", [])
    for pdef in provider_defs:
        name = pdef.get("name", "")
        if name not in _PROVIDER_CLASSES:
            logger.warning("Unknown provider '%s' — skipping", name)
            continue
        if not pdef.get("enabled", True):
            continue

        # Resolve API key: explicit value or env var
        api_key = pdef.get("api_key", "")
        if not api_key:
            env_var = pdef.get("api_key_env", "")
            if env_var:
                api_key = os.getenv(env_var, "")

        cls = _PROVIDER_CLASSES[name]
        provider = cls(api_key=api_key, base_url=pdef.get("base_url", ""))
        registry.register(provider)

        # Register explicit models
        for mdef in pdef.get("models", []):
            registry.register_model(mdef.get("name", ""), name)

        # Set up fallback chain
        fallbacks = pdef.get("fallback", [])
        if fallbacks:
            registry.set_fallbacks(name, fallbacks)

        logger.info("Provider '%s' initialised with %d models", name, len(pdef.get("models", [])))

    init_registry(registry)
    total = sum(len(p.get("models", [])) for p in provider_defs)
    logger.info("Provider registry initialised: %d providers, %d models", len(provider_defs), total)


def _init_observe() -> None:
    """Initialise the AI observability system (traces, scoring, quality tracking)."""
    from llmstack.observe._state import init_observe
    from llmstack.observe.traces import TraceStore
    from llmstack.observe.scoring import QualityScorer
    from llmstack.observe.tracker import QualityTracker
    from llmstack.observe.ab_testing import ABTestManager

    init_observe(
        trace_store=TraceStore(max_size=5000),
        scorer=QualityScorer(),
        tracker=QualityTracker(),
        ab_manager=ABTestManager(),
    )
    logger.info("AI observability initialised: traces, scoring, quality tracking, A/B testing")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect cache, init router, providers, observe. Shutdown: close connections."""
    from llmstack.gateway.cache import get_cache
    cache = await get_cache()
    _init_router()
    _init_providers()
    _init_observe()
    yield
    # Graceful shutdown: wait for in-flight requests to drain
    from llmstack.gateway.middleware.metrics import get_active_requests

    deadline = time.monotonic() + 30  # 30 s drain timeout
    while get_active_requests() > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.5)
    # Close persistent connection pool
    from llmstack.gateway.proxy import close_pool

    await close_pool()
    await cache.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLMStack Gateway",
        summary="OpenAI-compatible API gateway for local and cloud LLMs",
        description=(
            "LLMStack Gateway is a drop-in replacement for the OpenAI API that adds "
            "smart model routing, semantic caching, RAG, observability, and "
            "multi-provider resilience on top of Ollama, OpenAI, Anthropic, Google, "
            "and other LLM providers.\n\n"
            "## Key Features\n\n"
            "- **Smart Routing** — automatically pick the cheapest model that can handle each request\n"
            "- **Semantic Caching** — deduplicate similar prompts to cut costs and latency\n"
            "- **RAG Pipeline** — ingest documents and query them with retrieval-augmented generation\n"
            "- **Multi-Provider** — unified API across Ollama, OpenAI, Anthropic, Google, Groq, and more\n"
            "- **Observability** — traces, quality scoring, A/B testing, cost tracking, and alerts\n"
            "- **Batch Processing** — fan out requests in parallel for bulk workloads\n\n"
            "Works with any OpenAI-compatible SDK or client library."
        ),
        version="1.0.0",
        lifespan=lifespan,
        contact={
            "name": "LLMStack",
            "url": "https://github.com/mara-werils/llmstack",
        },
        license_info={
            "name": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        openapi_tags=[
            {
                "name": "Chat",
                "description": "OpenAI-compatible chat completions with streaming support. "
                "Supports smart model routing and semantic caching.",
            },
            {
                "name": "Embeddings",
                "description": "Generate vector embeddings for text using local or cloud models.",
            },
            {
                "name": "Models",
                "description": "List and inspect available models across all configured providers.",
            },
            {
                "name": "RAG",
                "description": "Document ingestion, chunking, and semantic search for "
                "retrieval-augmented generation workflows.",
            },
            {
                "name": "Observe",
                "description": "AI observability: request traces, quality scoring, drift alerts, "
                "and A/B test management.",
            },
            {
                "name": "Learn",
                "description": "Adaptive learning pipeline: collect feedback and fine-tune "
                "model behavior over time.",
            },
            {
                "name": "Templates",
                "description": "Create, version, and render reusable prompt templates with variables.",
            },
            {
                "name": "Conversations",
                "description": "Persistent, multi-turn conversation history with search and export.",
            },
            {
                "name": "Cost",
                "description": "Real-time cost tracking, budget enforcement, and savings reports.",
            },
            {
                "name": "Webhooks",
                "description": "Event-driven notifications for cost alerts, quality drift, and errors.",
            },
            {
                "name": "Batch",
                "description": "Submit and manage batch jobs for parallel request processing.",
            },
            {
                "name": "Leaderboard",
                "description": "Compare model performance across latency, quality, and cost metrics.",
            },
            {
                "name": "Router",
                "description": "Smart model routing statistics, decision logs, and strategy configuration.",
            },
            {
                "name": "Health",
                "description": "Liveness and readiness probes, Prometheus-compatible metrics endpoint.",
            },
        ],
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    cors_origins = os.getenv("LLMSTACK_CORS_ORIGINS", "*").split(",")
    if cors_origins == ["*"] and os.getenv("LLMSTACK_ENV", "") == "production":
        logger.warning(
            "CORS is set to allow all origins ('*') in production mode. "
            "Set LLMSTACK_CORS_ORIGINS to restrict allowed origins."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GZip compression for responses > 500 bytes
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # Middleware stack (order matters: outermost first)
    # 0. Request size limit (reject oversized payloads before processing)
    app.add_middleware(RequestSizeMiddleware, max_bytes=10 * 1024 * 1024)

    # 0.5. Correlation ID (assign unique ID to every request for tracing)
    app.add_middleware(CorrelationMiddleware)

    # 1. Logging (outermost — captures everything)
    app.add_middleware(LoggingMiddleware)

    # 2. Auth
    api_keys = os.getenv("LLMSTACK_API_KEYS", "")
    if api_keys:
        app.add_middleware(AuthMiddleware, api_keys=api_keys.split(","))

    # 3. Rate limiting
    rate_limit = os.getenv("LLMSTACK_RATE_LIMIT", "100/min")
    app.add_middleware(RateLimitMiddleware, rate_limit=rate_limit)

    # 4. Metrics (innermost — measures actual handler time)
    app.add_middleware(MetricsMiddleware)

    # Routes
    app.include_router(chat_router, prefix="/v1")
    app.include_router(embeddings_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(rag_router, prefix="/v1")
    app.include_router(router_router, prefix="/v1")
    app.include_router(observe_router, prefix="/v1")
    app.include_router(learn_router, prefix="/v1")
    app.include_router(templates_router, prefix="/v1")
    app.include_router(conversations_router, prefix="/v1")
    app.include_router(cost_router, prefix="/v1")
    app.include_router(webhooks_router, prefix="/v1")
    app.include_router(batch_router, prefix="/v1")
    app.include_router(leaderboard_router, prefix="/v1")
    app.include_router(health_router)
    app.include_router(widget_router)

    # Serve Web UI
    if _UI_DIR.is_dir():
        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(_UI_DIR / "index.html", media_type="text/html")

        app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")

    return app


app = create_app()
