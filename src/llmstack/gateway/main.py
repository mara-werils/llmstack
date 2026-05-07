"""LLMStack Gateway — OpenAI-compatible API gateway with caching, RAG, and resilience."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llmstack.gateway.routes.chat import router as chat_router
from llmstack.gateway.routes.embeddings import router as embeddings_router
from llmstack.gateway.routes.models import router as models_router
from llmstack.gateway.routes.health import router as health_router
from llmstack.gateway.routes.rag import router as rag_router
from llmstack.gateway.middleware.auth import AuthMiddleware
from llmstack.gateway.middleware.metrics import MetricsMiddleware
from llmstack.gateway.middleware.rate_limit import RateLimitMiddleware
from llmstack.gateway.middleware.logging import LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect cache. Shutdown: close connections."""
    from llmstack.gateway.cache import get_cache
    cache = await get_cache()
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
    app.include_router(health_router)

    return app


app = create_app()
