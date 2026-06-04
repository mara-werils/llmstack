"""Comprehensive tests for hybrid search — BM25, vector search, and RRF fusion."""

from __future__ import annotations

import numpy as np
import pytest

from llmstack.ask.hybrid_search import BM25, HybridSearcher, _tokenize
from llmstack.ask.parsers import TextChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks() -> list[TextChunk]:
    return [
        TextChunk(
            content="Python is a programming language for data science.",
            source="a.py",
            start_line=1,
            end_line=1,
        ),
        TextChunk(
            content="JavaScript runs in the browser and server.",
            source="b.js",
            start_line=1,
            end_line=1,
        ),
        TextChunk(
            content="Rust provides memory safety without garbage collection.",
            source="c.rs",
            start_line=1,
            end_line=1,
        ),
        TextChunk(
            content="Python web frameworks include Django and Flask.",
            source="d.py",
            start_line=1,
            end_line=1,
        ),
        TextChunk(
            content="Docker containers package applications for deployment.",
            source="e.md",
            start_line=1,
            end_line=1,
        ),
    ]


@pytest.fixture
def bm25(chunks: list[TextChunk]) -> BM25:
    b = BM25()
    b.index(chunks)
    return b


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_words(self) -> None:
        tokens = _tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercased(self) -> None:
        tokens = _tokenize("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_snake_case_splitting(self) -> None:
        tokens = _tokenize("my_variable_name")
        assert "my_variable_name" in tokens
        assert "variable" in tokens
        assert "name" in tokens

    def test_short_parts_excluded(self) -> None:
        tokens = _tokenize("a_b_longpart")
        # "a" and "b" are less than 2 chars, should be excluded from split parts
        assert "longpart" in tokens
        assert "a" not in [t for t in tokens if t != "a_b_longpart"]

    def test_numbers_in_identifiers(self) -> None:
        tokens = _tokenize("var1 var2")
        assert "var1" in tokens
        assert "var2" in tokens

    def test_punctuation_removed(self) -> None:
        tokens = _tokenize("hello, world! foo.bar")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_input(self) -> None:
        tokens = _tokenize("")
        assert tokens == []

    def test_only_punctuation(self) -> None:
        tokens = _tokenize("!@#$%^&*()")
        assert tokens == []


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class TestBM25:
    def test_index_sets_corpus(self, bm25: BM25, chunks: list[TextChunk]) -> None:
        assert bm25._n_docs == len(chunks)
        assert bm25._avg_dl > 0

    def test_search_returns_relevant_docs(self, bm25: BM25) -> None:
        results = bm25.search("Python programming")
        assert len(results) >= 1
        # The Python chunks should rank highest
        top_indices = [idx for idx, _score in results[:2]]
        assert 0 in top_indices or 3 in top_indices  # chunks about Python

    def test_search_returns_scores_descending(self, bm25: BM25) -> None:
        results = bm25.search("programming language")
        if len(results) > 1:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, bm25: BM25) -> None:
        results = bm25.search("")
        assert results == []

    def test_search_no_match(self, bm25: BM25) -> None:
        results = bm25.search("xyznonexistent")
        assert results == []

    def test_search_respects_top_k(self, bm25: BM25) -> None:
        results = bm25.search("Python", top_k=1)
        assert len(results) <= 1

    def test_search_on_empty_corpus(self) -> None:
        b = BM25()
        b.index([])
        results = b.search("anything")
        assert results == []

    def test_custom_k1_and_b(self, chunks: list[TextChunk]) -> None:
        b = BM25(k1=2.0, b=0.5)
        b.index(chunks)
        results = b.search("Python")
        assert len(results) >= 1

    def test_score_doc_no_match(self, bm25: BM25) -> None:
        score = bm25._score_doc(["nonexistent"], ["hello", "world"])
        assert score == 0.0

    def test_single_doc_single_term(self) -> None:
        b = BM25()
        b.index([TextChunk(content="hello hello hello", source="f.py", start_line=1, end_line=1)])
        results = b.search("hello")
        assert len(results) == 1
        assert results[0][1] > 0


# ---------------------------------------------------------------------------
# HybridSearcher
# ---------------------------------------------------------------------------


class TestHybridSearcher:
    def test_search_empty_chunks(self) -> None:
        hs = HybridSearcher()
        results = hs.search("anything")
        assert results == []

    def test_bm25_only_no_embeddings(self, chunks: list[TextChunk]) -> None:
        hs = HybridSearcher()
        hs.index(chunks)
        results = hs.search("Python programming")
        assert len(results) >= 1
        # Results should be TextChunk objects
        assert all(isinstance(chunk, TextChunk) for chunk, _ in results)

    def test_hybrid_with_embeddings(self, chunks: list[TextChunk]) -> None:
        dim = 8
        embeddings = np.random.randn(len(chunks), dim).astype(np.float32)
        query_emb = np.random.randn(dim).astype(np.float32)

        hs = HybridSearcher()
        hs.index(chunks, embeddings=embeddings)
        results = hs.search("Python", query_embedding=query_emb)
        assert len(results) >= 1

    def test_top_k_respected(self, chunks: list[TextChunk]) -> None:
        hs = HybridSearcher()
        hs.index(chunks)
        results = hs.search("Python", top_k=2)
        assert len(results) <= 2

    def test_rrf_scores_are_positive(self, chunks: list[TextChunk]) -> None:
        hs = HybridSearcher()
        hs.index(chunks)
        results = hs.search("Python")
        for _, score in results:
            assert score > 0

    def test_custom_weights(self, chunks: list[TextChunk]) -> None:
        hs = HybridSearcher(bm25_weight=0.9, vector_weight=0.1)
        hs.index(chunks)
        results = hs.search("Python")
        assert len(results) >= 1

    def test_scores_sorted_descending(self, chunks: list[TextChunk]) -> None:
        hs = HybridSearcher()
        hs.index(chunks)
        results = hs.search("programming language")
        if len(results) > 1:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Vector search (internal)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_vector_search_returns_top_k(self) -> None:
        hs = HybridSearcher()
        embeddings = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0.9, 0.1, 0, 0],
            ],
            dtype=np.float32,
        )
        hs._embeddings = embeddings

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = hs._vector_search(query, top_k=2)
        assert len(results) == 2
        # Index 0 should be most similar to query [1,0,0,0]
        assert results[0][0] == 0

    def test_vector_search_zero_query(self) -> None:
        hs = HybridSearcher()
        hs._embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)
        query = np.array([0, 0], dtype=np.float32)
        results = hs._vector_search(query, top_k=2)
        assert results == []

    def test_vector_search_no_embeddings(self) -> None:
        hs = HybridSearcher()
        hs._embeddings = None
        results = hs._vector_search(np.array([1, 0]), top_k=2)
        assert results == []

    def test_vector_search_empty_embeddings(self) -> None:
        hs = HybridSearcher()
        hs._embeddings = np.array([]).reshape(0, 4)
        results = hs._vector_search(np.array([1, 0, 0, 0]), top_k=2)
        assert results == []
