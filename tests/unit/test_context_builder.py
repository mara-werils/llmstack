"""Tests for smart context builder."""

import subprocess
from pathlib import Path

import pytest
from llmstack.context.builder import ContextBuilder, ContextChunk


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
    (tmp_path / "server.py").write_text("""
import main

def start_server(config):
    port = config["port"]
    print(f"Starting on {port}")
    return {"running": True}
""")
    (tmp_path / "utils.py").write_text("""
def helper():
    return "hello"

def format_output(data):
    return str(data)
""")
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


def test_context_chunk_properties():
    chunk = ContextChunk(
        file="a.py",
        content="x",
        relevance=0.05,
        reason="r",
        line_start=5,
        line_end=10,
        tokens_estimate=1,
    )
    assert chunk.line_count == 6
    assert chunk.is_relevant is False
    chunk.relevance = 0.5
    assert chunk.is_relevant is True


def _flaky_read_text(target_name: str):
    """Return a Path.read_text replacement that raises OSError for one filename."""
    original = Path.read_text

    def _flaky(self, *args, **kwargs):
        if self.name == target_name:
            raise OSError("boom")
        return original(self, *args, **kwargs)

    return _flaky


def _git_commit_all(repo_dir, message):
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "add", "-A"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-m", message],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )


def test_smart_context_skips_unreadable_file(project, monkeypatch):
    monkeypatch.setattr(Path, "read_text", _flaky_read_text("utils.py"))
    builder = ContextBuilder(project, max_tokens=5000)
    chunks = builder.build("helper", strategy="smart")
    assert all(c.file != "utils.py" for c in chunks)


def test_find_best_section_picks_highest_scoring_window(tmp_path):
    lines = [f"filler line {i}" for i in range(80)]
    lines[60] = "keyword_target here"
    (tmp_path / "big.py").write_text("\n".join(lines))
    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("keyword_target", strategy="smart")
    assert len(chunks) == 1
    assert chunks[0].line_start <= 61 <= chunks[0].line_end


def test_fit_budget_truncates_when_remaining_room(tmp_path):
    builder = ContextBuilder(tmp_path, max_tokens=160)
    chunk1 = ContextChunk(
        file="a.py",
        content="x" * 200,
        relevance=0.9,
        reason="r",
        line_start=1,
        line_end=10,
        tokens_estimate=50,
    )
    chunk2 = ContextChunk(
        file="b.py",
        content="y" * 2000,
        relevance=0.5,
        reason="r",
        line_start=1,
        line_end=10,
        tokens_estimate=500,
    )
    result = builder._fit_budget([chunk1, chunk2])
    assert result == [chunk1, chunk2]
    assert chunk2.tokens_estimate == 110
    assert len(chunk2.content) == 440


def test_git_context_recently_modified_files(tmp_path):
    (tmp_path / "main.py").write_text("def main():\n    server_call()\n")
    (tmp_path / "README.md").write_text("# readme mentions server too")
    _git_commit_all(tmp_path, "init")

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="git")

    files = {c.file for c in chunks}
    assert "main.py" in files
    assert "README.md" not in files  # not a recognised code extension


def test_git_context_skips_deleted_files(tmp_path):
    f = tmp_path / "temp.py"
    f.write_text("server placeholder\n")
    _git_commit_all(tmp_path, "add")
    f.unlink()
    _git_commit_all(tmp_path, "remove")

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="git")
    assert chunks == []


def test_git_context_skips_unreadable_file(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("server code\n")
    _git_commit_all(tmp_path, "init")
    monkeypatch.setattr(Path, "read_text", _flaky_read_text("main.py"))

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="git")
    assert chunks == []


def test_git_context_not_a_repo_returns_empty(tmp_path):
    (tmp_path / "main.py").write_text("server code\n")
    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="git")
    assert chunks == []


def test_git_context_handles_subprocess_exception(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("server code\n")

    def boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)
    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="git")
    assert chunks == []


def test_git_context_skips_blank_filename_entries(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("server code\n")

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="main.py\n\nmissing.py\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_result)

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="git")
    assert {c.file for c in chunks} == {"main.py"}


def test_import_context_handles_from_import(tmp_path):
    (tmp_path / "app.py").write_text("from helpers import run\nrun()\n")
    (tmp_path / "helpers.py").write_text("def run():\n    pass\n")

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("app", strategy="imports")

    assert any(c.file == "helpers.py" for c in chunks)


def test_import_context_follows_imports(tmp_path):
    (tmp_path / "app.py").write_text("import helpers\nhelpers.run()\n")
    (tmp_path / "helpers.py").write_text("def run():\n    pass\n")
    (tmp_path / "unrelated.py").write_text("x = 1\n")

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("app", strategy="imports")

    files = {c.file for c in chunks}
    assert "helpers.py" in files
    assert "unrelated.py" not in files


def test_import_context_no_relevant_file_returns_empty(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("zzz_no_match_anywhere", strategy="imports")
    assert chunks == []


def test_import_context_syntax_error_returns_empty(tmp_path):
    (tmp_path / "app.py").write_text("def broken(:\n    pass\n")
    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("app", strategy="imports")
    assert chunks == []


def test_import_context_skips_unreadable_matched_file(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("import helpers\n")
    (tmp_path / "helpers.py").write_text("def run(): pass\n")
    monkeypatch.setattr(Path, "read_text", _flaky_read_text("helpers.py"))

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("app", strategy="imports")
    assert all(c.file != "helpers.py" for c in chunks)


def test_related_context_skips_unreadable_file(tmp_path, monkeypatch):
    (tmp_path / "server_x.py").write_text("content\n")
    monkeypatch.setattr(Path, "read_text", _flaky_read_text("server_x.py"))

    builder = ContextBuilder(tmp_path, max_tokens=5000)
    chunks = builder.build("server", strategy="related")
    assert chunks == []
