"""Hybrid search — combines BM25 keyword search with vector cosine similarity.

Better recall than either method alone:
- BM25 catches exact keyword matches (function names, variable names, error messages)
- Vector search catches semantic similarity (meaning, intent, paraphrasing)

Results are combined using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from llmstack.ask.parsers import TextChunk


class BM25:
    """BM25 keyword ranking over text chunks.

    Lightweight pure-Python implementation (no external deps).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def index(self, chunks: list[TextChunk]) -> None:
        """Build the BM25 index from chunks."""
        self._corpus = []
        self._doc_freqs = defaultdict(int)

        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            self._corpus.append(tokens)
            unique = set(tokens)
            for term in unique:
                self._doc_freqs[term] += 1

        self._n_docs = len(self._corpus)
        total_tokens = sum(len(doc) for doc in self._corpus)
        self._avg_dl = total_tokens / max(self._n_docs, 1)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search and return (doc_index, score) pairs, sorted by score descending."""
        query_tokens = _tokenize(query)
        if not query_tokens or not self._corpus:
            return []

        scores: list[tuple[int, float]] = []

        for idx, doc_tokens in enumerate(self._corpus):
            score = self._score_doc(query_tokens, doc_tokens)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score_doc(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """Compute BM25 score for a single document."""
        dl = len(doc_tokens)
        score = 0.0

        # Term frequencies in this document
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1

        for term in query_tokens:
            if term not in tf_map:
                continue

            tf = tf_map[term]
            df = self._doc_freqs.get(term, 0)

            # IDF with smoothing
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 TF normalization
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
            score += idf * numerator / denominator

        return score


class HybridSearcher:
    """Combines BM25 and vector search using Reciprocal Rank Fusion.

    RRF formula: score = sum(1 / (k + rank_i)) for each retrieval method
    where k is a constant (default 60) that controls the influence of high vs low ranks.
    """

    def __init__(
        self,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self._bm25 = BM25()
        self._chunks: list[TextChunk] = []
        self._embeddings: np.ndarray | None = None

    def index(self, chunks: list[TextChunk], embeddings: np.ndarray | None = None) -> None:
        """Build both BM25 and vector indices."""
        self._chunks = chunks
        self._embeddings = embeddings
        self._bm25.index(chunks)

    def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[tuple[TextChunk, float]]:
        """Hybrid search combining BM25 and vector similarity.

        Returns (chunk, combined_score) pairs sorted by relevance.
        """
        if not self._chunks:
            return []

        rrf_scores: dict[int, float] = defaultdict(float)

        # BM25 retrieval
        bm25_results = self._bm25.search(query, top_k=top_k * 3)
        for rank, (idx, _score) in enumerate(bm25_results):
            rrf_scores[idx] += self.bm25_weight / (self.rrf_k + rank + 1)

        # Vector retrieval
        if (
            query_embedding is not None
            and self._embeddings is not None
            and self._embeddings.size > 0
        ):
            vector_results = self._vector_search(query_embedding, top_k * 3)
            for rank, (idx, _score) in enumerate(vector_results):
                rrf_scores[idx] += self.vector_weight / (self.rrf_k + rank + 1)

        # Sort by combined RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self._chunks[idx], score) for idx, score in ranked]

    def _vector_search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Cosine similarity search against stored embeddings."""
        emb = self._embeddings
        if emb is None or emb.size == 0:
            return []

        # Normalize
        q = query_emb.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []
        q = q / q_norm

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normed = emb / norms

        similarities = normed @ q
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z_][a-z0-9_]*\b", text)
    # Also split camelCase and snake_case for better code matching
    expanded: list[str] = []
    for t in tokens:
        expanded.append(t)
        # Split snake_case
        if "_" in t:
            expanded.extend(p for p in t.split("_") if len(p) >= 2)
        # Split camelCase
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", t)
        if len(parts) > 1:
            expanded.extend(p.lower() for p in parts if len(p) >= 2)
    return expanded
