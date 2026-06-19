"""Tests for Ask v2 — persistent index, AST chunking, hybrid search, git context, conversation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from llmstack.ask.index import PersistentIndex, _file_hash
from llmstack.ask.ast_chunker import chunk_python, chunk_code, _fallback_chunk
from llmstack.ask.hybrid_search import BM25, HybridSearcher, _tokenize
from llmstack.ask.git_context import get_git_info, GitInfo
from llmstack.ask.conversation import ConversationEngine, ConversationTurn
from llmstack.ask.parsers import TextChunk


# ===================================================================
# Persistent Index tests
# ===================================================================


class TestPersistentIndex:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")
        return tmp_path

    def test_create_index(self, project):
        idx = PersistentIndex(project)
        assert not idx.exists()
        idx._ensure_db()
        assert idx.exists()
        idx.close()

    def test_foreign_keys_enabled(self, project):
        idx = PersistentIndex(project)
        conn = idx._ensure_db()
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        idx.close()

    def test_delete_file_cascades_to_chunks(self, project):
        idx = PersistentIndex(project)
        conn = idx._ensure_db()
        conn.execute("INSERT INTO files (path, hash, mtime) VALUES ('x.py', 'h', 1.0)")
        conn.execute(
            "INSERT INTO chunks (file_path, content, start_line, end_line, chunk_index) "
            "VALUES ('x.py', 'c', 1, 1, 0)"
        )
        conn.commit()
        conn.execute("DELETE FROM files WHERE path='x.py'")
        conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path='x.py'").fetchone()[0]
        assert remaining == 0
        idx.close()

    def test_diff_all_new(self, project):
        idx = PersistentIndex(project)
        files = [project / "a.py", project / "b.py"]
        to_update, _, removed = idx.diff(files)
        assert len(to_update) == 2
        assert len(removed) == 0
        idx.close()

    def test_diff_unchanged(self, project):
        idx = PersistentIndex(project)
        files = [project / "a.py", project / "b.py"]

        # First index
        chunks = {
            "a.py": [TextChunk(content="def foo(): pass", source="a.py", start_line=1, end_line=1)],
            "b.py": [TextChunk(content="def bar(): pass", source="b.py", start_line=1, end_line=1)],
        }
        hashes = {
            "a.py": _file_hash(project / "a.py"),
            "b.py": _file_hash(project / "b.py"),
        }
        all_chunks = chunks["a.py"] + chunks["b.py"]
        idx.update(chunks, None, hashes, all_chunks)

        # Second diff — nothing changed
        to_update, _, removed = idx.diff(files)
        assert len(to_update) == 0
        assert len(removed) == 0
        idx.close()

    def test_diff_modified(self, project):
        idx = PersistentIndex(project)
        files = [project / "a.py", project / "b.py"]

        chunks = {"a.py": [TextChunk("x", "a.py", 1, 1)]}
        hashes = {"a.py": _file_hash(project / "a.py")}
        idx.update(chunks, None, hashes, chunks["a.py"])

        # Modify a.py
        (project / "a.py").write_text("def foo_modified(): pass\n")

        to_update, _, _ = idx.diff(files)
        modified = [str(p.name) for p in to_update]
        assert "a.py" in modified
        idx.close()

    def test_diff_removed(self, project):
        idx = PersistentIndex(project)
        chunks = {"a.py": [TextChunk("x", "a.py", 1, 1)], "b.py": [TextChunk("y", "b.py", 1, 1)]}
        hashes = {"a.py": _file_hash(project / "a.py"), "b.py": _file_hash(project / "b.py")}
        idx.update(chunks, None, hashes, chunks["a.py"] + chunks["b.py"])

        # Only a.py exists now
        to_update, _, removed = idx.diff([project / "a.py"])
        assert "b.py" in removed
        idx.close()

    def test_load_chunks(self, project):
        idx = PersistentIndex(project)
        chunks = {
            "a.py": [
                TextChunk("line1", "a.py", 1, 1),
                TextChunk("line2", "a.py", 2, 2),
            ]
        }
        idx.update(chunks, None, {"a.py": "hash"}, chunks["a.py"])
        loaded = idx.load_chunks()
        assert len(loaded) == 2
        assert loaded[0].content == "line1"
        idx.close()

    def test_save_and_load_embeddings(self, project):
        idx = PersistentIndex(project)
        emb = np.random.rand(5, 128).astype(np.float32)
        idx.update(
            {"a.py": [TextChunk("x", "a.py", 1, 1)]},
            emb,
            {"a.py": "h"},
            [TextChunk("x", "a.py", 1, 1)],
        )
        loaded = idx.load_embeddings()
        assert loaded is not None
        assert loaded.shape == (5, 128)
        idx.close()

    def test_clear(self, project):
        idx = PersistentIndex(project)
        idx.update(
            {"a.py": [TextChunk("x", "a.py", 1, 1)]},
            None,
            {"a.py": "h"},
            [TextChunk("x", "a.py", 1, 1)],
        )
        assert idx.chunk_count() == 1
        idx.clear()
        assert idx.chunk_count() == 0
        idx.close()

    def test_file_count(self, project):
        idx = PersistentIndex(project)
        idx.update(
            {"a.py": [TextChunk("x", "a.py", 1, 1)], "b.py": [TextChunk("y", "b.py", 1, 1)]},
            None,
            {"a.py": "h1", "b.py": "h2"},
            [TextChunk("x", "a.py", 1, 1), TextChunk("y", "b.py", 1, 1)],
        )
        assert idx.file_count() == 2
        idx.close()


class TestFileHash:
    def test_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_hash_changes_on_modify(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("version 1")
        h1 = _file_hash(f)
        f.write_text("version 2")
        h2 = _file_hash(f)
        assert h1 != h2

    def test_hash_nonexistent(self, tmp_path):
        assert _file_hash(tmp_path / "nonexistent") == ""


# ===================================================================
# AST Chunker tests
# ===================================================================


class TestPythonASTChunker:
    def test_splits_functions(self):
        source = textwrap.dedent("""\
            import os

            def foo():
                return 1

            def bar():
                return 2
        """)
        chunks = chunk_python(source, "test.py")
        names = [c.content for c in chunks]
        assert any("foo" in n for n in names)
        assert any("bar" in n for n in names)

    def test_splits_classes(self):
        source = textwrap.dedent("""\
            class MyClass:
                def method_a(self):
                    pass

                def method_b(self):
                    pass
        """)
        chunks = chunk_python(source, "test.py")
        assert any("MyClass" in c.content for c in chunks)

    def test_large_class_splits_methods(self):
        methods = "\n".join(
            f"    def method_{i}(self):\n" + "        pass\n" * 5 for i in range(20)
        )
        source = f"class BigClass:\n{methods}"
        chunks = chunk_python(source, "test.py")
        # Should have both class chunk and method chunks
        assert len(chunks) > 2

    def test_preserves_line_numbers(self):
        source = "import os\n\ndef foo():\n    return 1\n"
        chunks = chunk_python(source, "test.py")
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_handles_syntax_error(self):
        source = "def broken(:\n    pass"
        chunks = chunk_python(source, "test.py")
        # Should fallback gracefully
        assert len(chunks) > 0

    def test_module_header_captured(self):
        source = '"""Module docstring."""\nimport os\nimport sys\n\ndef foo(): pass\n'
        chunks = chunk_python(source, "test.py")
        assert any("import" in c.content for c in chunks)


class TestCodeChunker:
    def test_python_uses_ast(self):
        source = "def hello():\n    return 'world'\n"
        chunks = chunk_code(source, "test.py")
        assert len(chunks) > 0
        assert any("hello" in c.content for c in chunks)

    def test_javascript_splits(self):
        source = "function foo() {\n  return 1;\n}\n\nfunction bar() {\n  return 2;\n}\n"
        chunks = chunk_code(source, "test.js")
        assert len(chunks) >= 2

    def test_go_splits(self):
        source = "func Foo() int {\n  return 1\n}\n\nfunc Bar() int {\n  return 2\n}\n"
        chunks = chunk_code(source, "test.go")
        assert len(chunks) >= 2

    def test_unknown_ext_fallback(self):
        source = "line1\nline2\nline3\n"
        chunks = chunk_code(source, "test.xyz")
        assert len(chunks) > 0

    def test_fallback_chunk(self):
        lines = "\n".join(f"line {i}" for i in range(200))
        chunks = _fallback_chunk(lines, "test.txt", max_lines=80)
        assert len(chunks) >= 2


# ===================================================================
# BM25 tests
# ===================================================================


class TestBM25:
    @pytest.fixture
    def bm25(self):
        bm25 = BM25()
        chunks = [
            TextChunk("Python is a programming language for data science", "a.py", 1, 1),
            TextChunk("JavaScript is used for web development", "b.js", 1, 1),
            TextChunk("Rust provides memory safety without garbage collection", "c.rs", 1, 1),
            TextChunk("Python decorators are a powerful feature for metaprogramming", "d.py", 1, 1),
        ]
        bm25.index(chunks)
        return bm25

    def test_exact_match(self, bm25):
        results = bm25.search("Python programming language")
        assert len(results) > 0
        # First result should be about Python
        assert results[0][0] == 0 or results[0][0] == 3

    def test_no_match(self, bm25):
        results = bm25.search("quantum physics")
        assert len(results) == 0

    def test_partial_match(self, bm25):
        results = bm25.search("web development JavaScript")
        assert len(results) > 0

    def test_empty_query(self, bm25):
        assert bm25.search("") == []

    def test_empty_corpus(self):
        bm25 = BM25()
        bm25.index([])
        assert bm25.search("test") == []


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello world foo_bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo_bar" in tokens

    def test_snake_case_split(self):
        tokens = _tokenize("my_function_name")
        assert "my" in tokens
        assert "function" in tokens
        assert "name" in tokens

    def test_filters_short(self):
        tokens = _tokenize("a b c def")
        assert "def" in tokens
        # Single chars should be filtered


# ===================================================================
# Hybrid Search tests
# ===================================================================


class TestHybridSearcher:
    @pytest.fixture
    def searcher(self):
        chunks = [
            TextChunk("Authentication middleware validates API keys", "auth.py", 1, 5),
            TextChunk("Database connection pool management", "db.py", 1, 5),
            TextChunk("Rate limiting with token bucket algorithm", "rate.py", 1, 5),
            TextChunk("User authentication and authorization logic", "auth2.py", 1, 5),
        ]
        emb = np.random.rand(4, 64).astype(np.float32)
        s = HybridSearcher()
        s.index(chunks, emb)
        return s

    def test_bm25_only(self, searcher):
        results = searcher.search("authentication API keys", top_k=2)
        assert len(results) > 0

    def test_hybrid_with_embeddings(self, searcher):
        query_emb = np.random.rand(64).astype(np.float32)
        results = searcher.search("auth", query_embedding=query_emb, top_k=3)
        assert len(results) > 0

    def test_returns_chunks(self, searcher):
        results = searcher.search("database connection")
        for chunk, score in results:
            assert isinstance(chunk, TextChunk)
            assert isinstance(score, float)

    def test_empty_search(self):
        s = HybridSearcher()
        s.index([], None)
        results = s.search("test")
        assert results == []

    def test_top_k_limit(self, searcher):
        results = searcher.search("authentication", top_k=1)
        assert len(results) <= 1


# ===================================================================
# Git Context tests
# ===================================================================


class TestGitInfo:
    def test_to_context_no_repo(self):
        info = GitInfo(is_repo=False)
        assert info.to_context() == ""

    def test_to_context_with_data(self):
        info = GitInfo(
            is_repo=True,
            branch="main",
            recent_commits=[
                {"hash": "abc123", "author": "dev", "message": "fix bug", "when": "2h ago"},
            ],
            changed_files=["src/auth.py"],
        )
        ctx = info.to_context()
        assert "main" in ctx
        assert "fix bug" in ctx
        assert "auth.py" in ctx

    def test_to_context_empty_repo(self):
        info = GitInfo(is_repo=True, branch="main")
        ctx = info.to_context()
        assert "main" in ctx


class TestGetGitInfo:
    def test_real_repo(self):
        # This project IS a git repo
        info = get_git_info(Path(__file__).parent.parent.parent)
        assert info.is_repo
        assert info.branch

    def test_non_repo(self, tmp_path):
        info = get_git_info(tmp_path)
        assert not info.is_repo


# ===================================================================
# Conversation tests
# ===================================================================


class TestConversationTurn:
    def test_fields(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.sources == []


class TestConversationEngine:
    def test_init(self):
        engine = ConversationEngine()
        assert engine.turn_count == 0
        assert engine.history == []

    def test_clear(self):
        engine = ConversationEngine()
        engine._history.append(ConversationTurn(role="user", content="hi"))
        engine.clear()
        assert engine.turn_count == 0

    def test_build_messages(self):
        engine = ConversationEngine(model="test")
        engine._history.append(ConversationTurn(role="user", content="first question"))
        engine._history.append(ConversationTurn(role="assistant", content="first answer"))

        messages = engine._build_messages("second question")
        # system + history (2 turns) + current
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "first question"
        assert messages[2]["content"] == "first answer"
        assert messages[3]["content"] == "second question"

    def test_build_messages_with_git(self):
        engine = ConversationEngine(git_context="branch: main")
        messages = engine._build_messages("question")
        system = messages[0]["content"]
        assert "branch: main" in system

    def test_history_limit(self):
        engine = ConversationEngine()
        # Add 30 turns
        for i in range(30):
            engine._history.append(ConversationTurn(role="user", content=f"q{i}"))
            engine._history.append(ConversationTurn(role="assistant", content=f"a{i}"))

        messages = engine._build_messages("current")
        # system + 20 history (10 pairs) + current = 22
        assert len(messages) == 22


# ===================================================================
# Integration: combined features
# ===================================================================


class TestIndexWithASTChunker:
    def test_index_python_with_ast(self, tmp_path):
        code = textwrap.dedent("""\
            import os

            def process_data(items):
                return [x * 2 for x in items]

            class DataHandler:
                def __init__(self):
                    self.data = []

                def add(self, item):
                    self.data.append(item)
        """)
        (tmp_path / "handler.py").write_text(code)

        # AST chunk it
        chunks = chunk_code(code, "handler.py")
        assert len(chunks) >= 2

        # Index it
        idx = PersistentIndex(tmp_path)
        file_chunks = {"handler.py": chunks}
        idx.update(file_chunks, None, {"handler.py": "hash"}, chunks)
        loaded = idx.load_chunks()
        assert len(loaded) == len(chunks)
        idx.close()

    def test_hybrid_search_with_code(self):
        chunks = [
            TextChunk("def authenticate_user(token):\n    return verify(token)", "auth.py", 1, 2),
            TextChunk("def connect_database(url):\n    return Database(url)", "db.py", 1, 2),
            TextChunk("def rate_limit_check(key):\n    bucket = get_bucket(key)", "rate.py", 1, 2),
        ]
        searcher = HybridSearcher()
        searcher.index(chunks, None)

        results = searcher.search("authentication token verify")
        assert len(results) > 0
        assert "auth" in results[0][0].source
