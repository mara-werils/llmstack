"""Tests for RAG pipeline — context building and prompt construction."""

from __future__ import annotations


from llmstack.gateway.rag.pipeline import RAGPipeline
from llmstack.gateway.rag.store import SearchResult


class TestContextBuilding:
    """Test how search results are formatted into LLM context."""

    def test_single_result(self):
        results = [
            SearchResult(
                text="Python is a programming language.",
                score=0.92,
                metadata={"source": "docs.txt"},
            ),
        ]
        context = RAGPipeline._build_context(results)
        assert "docs.txt" in context
        assert "0.92" in context
        assert "Python is a programming language." in context

    def test_multiple_results_numbered(self):
        results = [
            SearchResult(text="First chunk.", score=0.9, metadata={"source": "a.txt"}),
            SearchResult(text="Second chunk.", score=0.8, metadata={"source": "b.txt"}),
            SearchResult(text="Third chunk.", score=0.7, metadata={"source": "c.txt"}),
        ]
        context = RAGPipeline._build_context(results)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context

    def test_results_separated(self):
        results = [
            SearchResult(text="A", score=0.9, metadata={"source": "x.txt"}),
            SearchResult(text="B", score=0.8, metadata={"source": "y.txt"}),
        ]
        context = RAGPipeline._build_context(results)
        assert "---" in context  # Separator between results

    def test_empty_results(self):
        context = RAGPipeline._build_context([])
        assert context == ""

    def test_unknown_source_handled(self):
        results = [
            SearchResult(text="No source.", score=0.5, metadata={}),
        ]
        context = RAGPipeline._build_context(results)
        assert "unknown" in context
