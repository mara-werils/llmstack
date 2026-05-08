"""LLMStack Gateway — OpenAI-compatible API gateway with caching, RAG, and resilience."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmstack.gateway.routes.chat import router as chat_router
from llmstack.gateway.routes.embeddings import router as embeddings_router
from llmstack.gateway.routes.models import router as models_router
from llmstack.gateway.routes.health import router as health_router
from llmstack.gateway.routes.rag import router as rag_router
from llmstack.gateway.routes.router import router as router_router
from llmstack.gateway.middleware.auth import AuthMiddleware
from llmstack.gateway.middleware.metrics import MetricsMiddleware
from llmstack.gateway.middleware.rate_limit import RateLimitMiddleware
from llmstack.gateway.middleware.logging import LoggingMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect cache, init router. Shutdown: close connections."""
    from llmstack.gateway.cache import get_cache
    cache = await get_cache()
    _init_router()
    yield
    await cache.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLMStack Gateway",
        description="OpenAI-compatible API gateway with caching, RAG, and resilience",
        version="0.3.0",
        lifespan=lifespan,
    )

    # CORS
    cors_origins = os.getenv("LLMSTACK_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware stack (order matters: outermost first)
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
    app.include_router(health_router)

    # Serve Web UI
    if _UI_DIR.is_dir():
        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(_UI_DIR / "index.html", media_type="text/html")

        app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")

    return app


app = create_app()
