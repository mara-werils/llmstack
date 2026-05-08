"""
FastAPI app powered by llmstack.

A production-ready API that uses llmstack as its LLM + RAG backend.
Demonstrates how to build a real application on top of llmstack's
OpenAI-compatible and RAG endpoints.

Install:
    pip install fastapi uvicorn httpx openai

Usage:
    1. Start llmstack:  llmstack init --preset rag && llmstack up
    2. Start this app:  uvicorn fastapi_app:app --port 3000 --reload
    3. Open docs:       http://localhost:3000/docs
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

# ── Configuration ─────────────────────────────────────────────────────
LLMSTACK_URL = "http://localhost:8000"
LLMSTACK_API_KEY = "llmstack"

# OpenAI client for chat/embeddings
llm = OpenAI(base_url=f"{LLMSTACK_URL}/v1", api_key=LLMSTACK_API_KEY)

# httpx client for RAG endpoints (not in OpenAI spec)
http = httpx.AsyncClient(
    base_url=LLMSTACK_URL,
    headers={"Authorization": f"Bearer {LLMSTACK_API_KEY}"},
    timeout=60.0,
)

app = FastAPI(
    title="Knowledge Base API",
    description="A knowledge base app backed by llmstack for inference and RAG.",
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────
class IngestRequest(BaseModel):
    """Upload a document to the knowledge base."""
    text: str
    source: str
    metadata: dict = {}


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    source: str


class AskRequest(BaseModel):
    """Ask a question against the knowledge base."""
    question: str
    top_k: int = 5
    model: str = "llama3.2"


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str


class SummarizeRequest(BaseModel):
    """Summarize a block of text."""
    text: str
    max_length: int = 200


class SummarizeResponse(BaseModel):
    summary: str
    model: str


class TranslateRequest(BaseModel):
    """Translate text to a target language."""
    text: str
    target_language: str
    source_language: str = "English"


class TranslateResponse(BaseModel):
    translation: str
    target_language: str


class HealthResponse(BaseModel):
    app: str
    llmstack: dict


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest):
    """Ingest a document into the knowledge base via llmstack RAG."""
    resp = await http.post("/v1/rag/ingest", json={
        "text": req.text,
        "source": req.source,
        "metadata": req.metadata,
    })

    if resp.status_code != 200:
        detail = resp.json().get("error", {}).get("message", "Ingestion failed")
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()
    return IngestResponse(
        status=data["status"],
        chunks_stored=data["chunks_stored"],
        source=data["source"],
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question and get an answer grounded in your documents."""
    resp = await http.post("/v1/rag/query", json={
        "question": req.question,
        "top_k": req.top_k,
        "model": req.model,
        "temperature": 0.1,
    })

    if resp.status_code != 200:
        detail = resp.json().get("error", {}).get("message", "Query failed")
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()
    return AskResponse(
        answer=data["answer"],
        sources=data.get("sources", []),
        model=data.get("model", req.model),
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(req: SummarizeRequest):
    """Summarize a block of text using the LLM."""
    response = llm.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Summarize the following text in {req.max_length} words or fewer. "
                    "Be concise and capture the key points."
                ),
            },
            {"role": "user", "content": req.text},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return SummarizeResponse(
        summary=response.choices[0].message.content,
        model=response.model,
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    """Translate text between languages using the LLM."""
    response = llm.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate the following {req.source_language} text to {req.target_language}. "
                    "Return only the translation, nothing else."
                ),
            },
            {"role": "user", "content": req.text},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return TranslateResponse(
        translation=response.choices[0].message.content,
        target_language=req.target_language,
    )


@app.delete("/documents/{source}")
async def delete_document(source: str):
    """Delete all chunks for a given source from the knowledge base."""
    resp = await http.delete(f"/v1/rag/documents/{source}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Delete failed")
    return resp.json()


@app.get("/rag/status")
async def rag_status():
    """Get statistics about the RAG knowledge base."""
    resp = await http.get("/v1/rag/status")
    return resp.json()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of this app and the llmstack backend."""
    try:
        resp = await http.get("/healthz")
        llmstack_health = resp.json()
    except Exception as exc:
        llmstack_health = {"status": "unreachable", "error": str(exc)}

    return HealthResponse(app="ok", llmstack=llmstack_health)


# ── Startup / shutdown ────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown():
    await http.aclose()


# ── Run directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=3000, reload=True)
