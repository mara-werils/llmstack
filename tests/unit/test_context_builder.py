"""Tests for smart context builder."""

import pytest
from llmstack.context.builder import ContextBuilder


@pytest.fixture
def project(tmp_path):
    (tmp_path / "main.py").write_text('''
def main():
    """Main entry point."""
    config = load_config()
    server = start_server(config)
    return server

def load_config():
    return {"port": 8000}
''')
    (tmp_path / "server.py").write_text('''
import main

def start_server(config):
    port = config["port"]
    print(f"Starting on {port}")
    return {"running": True}
''')
    (tmp_path / "utils.py").write_text('''
def helper():
    return "hello"

def format_output(data):
    return str(data)
''')
    (tmp_path / "README.md").write_text("# Test Project")
    return tmp_path


def test_smart_context(project):
    builder = ContextBuilder(project, max_tokens=5000)
    chunks = builder.build("server config", strategy="smart")
    assert len(chunks) > 0
    # Should find server.py and main.py as relevant
    files = {c.file for c in chunks}
    assert any("server" in f or "main" in f for f in files)


def test_related_context(project):
    builder = ContextBuilder(project, max_tokens=5000)
    chunks = builder.build("server", strategy="related")
    assert len(chunks) > 0
    files = {c.file for c in chunks}
    assert any("server" in f for f in files)


def test_token_budget(project):
    builder = ContextBuilder(project, max_tokens=100)
    chunks = builder.build("main server", strategy="smart")
    total_tokens = sum(c.tokens_estimate for c in chunks)
    assert total_tokens <= 100 + 50  # Allow some slack


def test_empty_query(project):
    builder = ContextBuilder(project, max_tokens=5000)
    chunks = builder.build("", strategy="smart")
    # Should return something even with empty query
    assert isinstance(chunks, list)


def test_no_matching_files(tmp_path):
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3")
    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("anything", strategy="smart")
    assert len(chunks) == 0


def test_relevance_scoring(project):
    builder = ContextBuilder(project, max_tokens=5000)
    chunks = builder.build("server", strategy="smart")
    if len(chunks) >= 2:
        # First chunk should have highest relevance
        assert chunks[0].relevance >= chunks[1].relevance
