"""LLMStack Gateway — OpenAI-compatible API gateway."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llmstack.gateway.routes.chat import router as chat_router
from llmstack.gateway.routes.embeddings import router as embeddings_router
from llmstack.gateway.routes.models import router as models_router
from llmstack.gateway.routes.health import router as health_router
from llmstack.gateway.middleware.auth import AuthMiddleware
from llmstack.gateway.middleware.metrics import MetricsMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLMStack Gateway",
        description="OpenAI-compatible API gateway for LLMStack",
        version="0.1.0",
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

    # Auth
    api_keys = os.getenv("LLMSTACK_API_KEYS", "")
    if api_keys:
        app.add_middleware(AuthMiddleware, api_keys=api_keys.split(","))

    # Metrics
    app.add_middleware(MetricsMiddleware)

    # Routes
    app.include_router(chat_router, prefix="/v1")
    app.include_router(embeddings_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(health_router)

    return app


app = create_app()
