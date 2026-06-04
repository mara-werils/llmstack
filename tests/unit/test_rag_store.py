"""Tests for RAG vector store — chunking and key generation."""

from __future__ import annotations


from llmstack.gateway.rag.store import VectorStore, CHUNK_SIZE


class TestChunking:
    """Test document chunking logic."""

    def setup_method(self):
        self.store = VectorStore(qdrant_url="http://fake:6333")

    def test_single_chunk(self):
        text = "Hello world this is a test"
        chunks = self.store._chunk_text(text, source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_multiple_chunks(self):
        # Create text longer than CHUNK_SIZE words
        words = [f"word{i}" for i in range(CHUNK_SIZE * 3)]
        text = " ".join(words)
        chunks = self.store._chunk_text(text, source="big.txt")
        assert len(chunks) > 1

        # Verify chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_overlap(self):
        """Consecutive chunks should overlap."""
        words = [f"word{i}" for i in range(CHUNK_SIZE * 2)]
        text = " ".join(words)
        chunks = self.store._chunk_text(text)

        if len(chunks) >= 2:
            # The step is CHUNK_SIZE - CHUNK_OVERLAP
            # So chunk 1 starts at (CHUNK_SIZE - CHUNK_OVERLAP)
            # The first CHUNK_OVERLAP words of chunk 1 should overlap with the end of chunk 0
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0

    def test_empty_text(self):
        chunks = self.store._chunk_text("")
        assert len(chunks) == 0

    def test_deterministic_ids(self):
        """Same text should produce the same chunk IDs."""
        text = "Deterministic chunking test"
        chunks1 = self.store._chunk_text(text, source="a.txt")
        chunks2 = self.store._chunk_text(text, source="a.txt")
        assert chunks1[0].id == chunks2[0].id

    def test_different_text_different_ids(self):
        chunks1 = self.store._chunk_text("hello world", source="a.txt")
        chunks2 = self.store._chunk_text("goodbye world", source="a.txt")
        assert chunks1[0].id != chunks2[0].id

    def test_metadata_includes_word_count(self):
        text = "one two three four five"
        chunks = self.store._chunk_text(text)
        assert chunks[0].metadata["word_count"] == 5


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_fields(self):
        from llmstack.gateway.rag.store import SearchResult

        result = SearchResult(text="hello", score=0.95, metadata={"source": "test.txt"})
        assert result.text == "hello"
        assert result.score == 0.95
        assert result.metadata["source"] == "test.txt"
