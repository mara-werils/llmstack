# ---------------------------------------------------------------------------
# Multi-stage production Dockerfile for llmstack gateway
# ---------------------------------------------------------------------------

# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --no-compile --prefix=/install ".[gateway]"

# ---- Final stage ----
FROM python:3.11-slim

LABEL maintainer="mara-werils"
LABEL org.opencontainers.image.source="https://github.com/mara-werils/llmstack"
LABEL org.opencontainers.image.description="LLMStack gateway - smart LLM routing and inference proxy"
LABEL org.opencontainers.image.version="1.0.0"

RUN groupadd --gid 1000 llmstack && \
    useradd --uid 1000 --gid llmstack --shell /bin/bash --create-home llmstack

COPY --from=builder /install /usr/local
COPY src/llmstack/gateway/ /app/llmstack/gateway/
COPY src/llmstack/__init__.py /app/llmstack/__init__.py

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER llmstack

CMD ["uvicorn", "llmstack.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
