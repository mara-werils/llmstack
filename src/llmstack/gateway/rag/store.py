"""Vector store client for RAG — wraps Qdrant operations.

Handles collection management, document chunking, embedding, and retrieval.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass

import httpx

QDRANT_URL = os.getenv("LLMSTACK_QDRANT_URL", "http://llmstack-qdrant:6333")
EMBEDDINGS_URL = os.getenv("LLMSTACK_EMBEDDINGS_URL", "")
INFERENCE_URL = os.getenv("LLMSTACK_INFERENCE_URL", "")

COLLECTION_NAME = "llmstack_documents"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class VectorStore:
    """Qdrant-backed vector store with embedding generation."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        embeddings_url: str = "",
    ):
        self._qdrant = qdrant_url
        self._embed_url = embeddings_url or EMBEDDINGS_URL or INFERENCE_URL
        self._dimension: int | None = None

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via the embeddings backend."""
        url = self._embed_url
        if not url.endswith("/embeddings"):
            url = f"{url}/embeddings"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                json={
                    "input": texts,
                    "model": "bge-m3",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        embeddings = [item["embedding"] for item in data["data"]]
        if not self._dimension and embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        # Probe dimension with a test embedding
        if not self._dimension:
            test = await self._embed(["test"])
            self._dimension = len(test[0])

        async with httpx.AsyncClient(timeout=10) as client:
            # Check if collection exists
            resp = await client.get(f"{self._qdrant}/collections/{COLLECTION_NAME}")
            if resp.status_code == 200:
                return

            # Create collection
            resp = await client.put(
                f"{self._qdrant}/collections/{COLLECTION_NAME}",
                json={
                    "vectors": {
                        "size": self._dimension,
                        "distance": "Cosine",
                    },
                    "optimizers_config": {
                        "indexing_threshold": 10000,
                    },
                },
            )
            resp.raise_for_status()

    def _chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks: list[Chunk] = []

        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i : i + CHUNK_SIZE]
            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash))

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": source,
                        "chunk_index": len(chunks),
                        "word_count": len(chunk_words),
                    },
                )
            )

        return chunks

    async def ingest(
        self,
        text: str,
        source: str = "",
        metadata: dict | None = None,
    ) -> int:
        """Chunk, embed, and store a document. Returns number of chunks stored."""
        await self.ensure_collection()

        chunks = self._chunk_text(text, source=source)
        if not chunks:
            return 0

        # Batch embed
        texts = [c.text for c in chunks]
        embeddings = await self._embed(texts)

        # Prepare Qdrant points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "text": chunk.text,
                "source": chunk.metadata.get("source", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                **({} if metadata is None else metadata),
            }
            points.append(
                {
                    "id": chunk.id,
                    "vector": embedding,
                    "payload": payload,
                }
            )

        # Upsert in batches of 100
        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(0, len(points), 100):
                batch = points[i : i + 100]
                resp = await client.put(
                    f"{self._qdrant}/collections/{COLLECTION_NAME}/points",
                    json={"points": batch},
                )
                resp.raise_for_status()

        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Semantic search over stored documents."""
        embeddings = await self._embed([query])
        query_vector = embeddings[0]

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self._qdrant}/collections/{COLLECTION_NAME}/points/search",
                json={
                    "vector": query_vector,
                    "limit": top_k,
                    "score_threshold": score_threshold,
                    "with_payload": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for hit in data.get("result", []):
            payload = hit.get("payload", {})
            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=hit.get("score", 0.0),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )

        return results

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self._qdrant}/collections/{COLLECTION_NAME}/points/delete",
                json={
                    "filter": {
                        "must": [
                            {"key": "source", "match": {"value": source}},
                        ],
                    },
                },
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("status", 0)

    async def collection_info(self) -> dict:
        """Get collection stats."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._qdrant}/collections/{COLLECTION_NAME}")
            if resp.status_code == 404:
                return {"status": "not_found", "points_count": 0}
            resp.raise_for_status()
            info = resp.json().get("result", {})
            return {
                "status": info.get("status", "unknown"),
                "points_count": info.get("points_count", 0),
                "vectors_count": info.get("vectors_count", 0),
                "segments_count": info.get("segments_count", 0),
            }


# Module-level singleton
_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
