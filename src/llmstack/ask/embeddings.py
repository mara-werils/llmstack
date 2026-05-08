"""In-memory vector search using numpy and Ollama embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import numpy as np

if TYPE_CHECKING:
    from llmstack.ask.parsers import TextChunk


class LocalEmbeddings:
    """Generate embeddings via Ollama API and search in-memory."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._embeddings: np.ndarray | None = None
        self._chunks: list[TextChunk] = []
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10))

    async def _ensure_model(self) -> None:
        """Pull the embedding model if it is not already available."""
        try:
            resp = await self._client.post(
                f"{self.ollama_url}/api/show",
                json={"name": self.model},
            )
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass

        # Model not found — pull it
        resp = await self._client.post(
            f"{self.ollama_url}/api/pull",
            json={"name": self.model, "stream": False},
            timeout=httpx.Timeout(600, connect=10),
        )
        resp.raise_for_status()

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Get embeddings from Ollama's /api/embed endpoint.

        Processes texts in batches to avoid overwhelming the API.
        Returns an (N, D) numpy array of float32 embeddings.
        """
        await self._ensure_model()

        batch_size = 32
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await self._client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    async def index(self, chunks: list[TextChunk]) -> None:
        """Embed and store all chunks for later search."""
        self._chunks = chunks
        texts = [c.content for c in chunks]
        self._embeddings = await self.embed(texts)

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[TextChunk, float]]:
        """Find most relevant chunks by cosine similarity.

        Returns a list of (chunk, similarity_score) tuples sorted by
        descending relevance.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query_emb = await self.embed([query])  # (1, D)
        query_vec = query_emb[0]

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)

        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1.0, norms)
        safe_query_norm = query_norm if query_norm != 0 else 1.0

        similarities = (self._embeddings @ query_vec) / (safe_norms * safe_query_norm)

        # Get top_k indices
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = [
            (self._chunks[i], float(similarities[i]))
            for i in top_indices
        ]
        return results

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
