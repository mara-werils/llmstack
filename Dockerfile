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

RUN pip install --no-cache-dir --prefix=/install ".[gateway]"

# ---- Final stage ----
FROM python:3.11-slim

LABEL maintainer="mara-werils"
LABEL org.opencontainers.image.source="https://github.com/mara-werils/llmstack"

RUN groupadd --gid 1000 llmstack && \
    useradd --uid 1000 --gid llmstack --shell /bin/bash --create-home llmstack

COPY --from=builder /install /usr/local
COPY src/llmstack/ /app/llmstack/

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER llmstack

CMD ["uvicorn", "llmstack.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
