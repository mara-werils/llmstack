FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    uvicorn[standard]>=0.30 \
    httpx>=0.27 \
    starlette>=0.40

COPY src/llmstack/__init__.py /app/llmstack/__init__.py
COPY src/llmstack/gateway/ /app/llmstack/gateway/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "llmstack.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
