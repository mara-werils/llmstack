"""RAG API endpoints — ingest documents and query with retrieval-augmented generation."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llmstack.gateway.rag.pipeline import RAGPipeline
from llmstack.gateway.rag.store import get_store

router = APIRouter()

INFERENCE_URL = os.getenv("LLMSTACK_INFERENCE_URL", "http://llmstack-ollama:11434/v1")


@router.post("/rag/ingest")
async def ingest(request: Request):
    """Ingest a document for RAG.

    Body:
        text: str — raw text content to ingest
        source: str — filename or URL for citation
        metadata: dict (optional) — extra metadata to store
    """
    body = await request.json()
    text = body.get("text", "")
    source = body.get("source", "unknown")
    metadata = body.get("metadata", {})

    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Field 'text' is required", "type": "validation_error"}},
        )

    if len(text) > 1_000_000:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Document too large (max 1MB text)",
                    "type": "validation_error",
                }
            },
        )

    store = get_store()
    chunks_count = await store.ingest(text=text, source=source, metadata=metadata)

    return JSONResponse(
        content={
            "status": "ok",
            "chunks_stored": chunks_count,
            "source": source,
        }
    )


@router.post("/rag/query")
async def query(request: Request):
    """Query documents with RAG.

    Body:
        question: str — the question to answer
        model: str (optional) — model to use for generation
        top_k: int (optional) — number of chunks to retrieve (default: 5)
        stream: bool (optional) — stream the response (default: false)
        temperature: float (optional) — generation temperature (default: 0.1)
        max_tokens: int (optional) — max tokens to generate (default: 1024)
    """
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse(
            status_code=400,
            content={
                "error": {"message": "Field 'question' is required", "type": "validation_error"}
            },
        )

    model = body.get("model", "llama3.2")
    top_k = body.get("top_k", 5)
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.1)
    max_tokens = body.get("max_tokens", 1024)

    pipeline = RAGPipeline(inference_url=INFERENCE_URL)

    if stream:
        return StreamingResponse(
            _stream_rag(pipeline, question, model, top_k, temperature, max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await pipeline.query(
        question=question,
        model=model,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return JSONResponse(
        content={
            "answer": result.answer,
            "sources": result.sources,
            "model": result.model,
            "usage": result.usage,
            "latency": result.latency,
        }
    )


async def _stream_rag(pipeline, question, model, top_k, temperature, max_tokens):
    """Generate SSE events for streaming RAG response."""
    async for chunk in pipeline.query_stream(
        question=question,
        model=model,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        if chunk.done:
            data = json.dumps({"done": True, "sources": chunk.sources or []})
            yield f"data: {data}\n\n"
        else:
            data = json.dumps({"token": chunk.token})
            yield f"data: {data}\n\n"


@router.delete("/rag/documents/{source}")
async def delete_document(source: str):
    """Delete all chunks from a specific source."""
    store = get_store()
    await store.delete_by_source(source)
    return JSONResponse(content={"status": "ok", "source": source})


@router.get("/rag/status")
async def rag_status():
    """Get RAG collection statistics."""
    store = get_store()
    info = await store.collection_info()
    return JSONResponse(content=info)
