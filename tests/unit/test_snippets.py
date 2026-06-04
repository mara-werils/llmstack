"""Tests for snippet manager."""

import pytest

from llmstack.snippets.manager import SnippetManager


@pytest.fixture
def manager(tmp_path):
    return SnippetManager(db_path=tmp_path / "test_snippets.db")


def test_save_and_get(manager):
    snippet = manager.save(
        title="Hello World",
        code='print("hello")',
        language="python",
        tags=["test", "example"],
        description="A simple test snippet",
    )
    assert snippet.id
    assert snippet.title == "Hello World"
    assert snippet.language == "python"
    assert "test" in snippet.tags

    retrieved = manager.get(snippet.id)
    assert retrieved is not None
    assert retrieved.title == snippet.title
    assert retrieved.code == snippet.code


def test_search_by_query(manager):
    manager.save(title="FastAPI Router", code="@app.get('/')", language="python", tags=["web"])
    manager.save(
        title="React Component", code="function App() {}", language="javascript", tags=["frontend"]
    )

    results = manager.search("FastAPI")
    assert len(results) >= 1
    assert results[0].title == "FastAPI Router"


def test_search_by_language(manager):
    manager.save(title="Py1", code="x = 1", language="python")
    manager.save(title="Js1", code="let x = 1", language="javascript")

    results = manager.search("", language="python")
    assert all(s.language == "python" for s in results)


def test_search_by_tag(manager):
    manager.save(title="Tagged", code="x = 1", tags=["important"])
    manager.save(title="Untagged", code="y = 2", tags=["other"])

    results = manager.search("", tag="important")
    assert len(results) == 1
    assert results[0].title == "Tagged"


def test_delete(manager):
    snippet = manager.save(title="To Delete", code="delete me")
    assert manager.delete(snippet.id)
    assert manager.get(snippet.id) is None


def test_count(manager):
    assert manager.count() == 0
    manager.save(title="S1", code="1")
    manager.save(title="S2", code="2")
    assert manager.count() == 2


def test_list_tags(manager):
    manager.save(title="A", code="a", tags=["python", "web"])
    manager.save(title="B", code="b", tags=["python", "cli"])

    tags = manager.list_tags()
    assert tags["python"] == 2
    assert tags["web"] == 1


def test_list_languages(manager):
    manager.save(title="Py", code="x", language="python")
    manager.save(title="Py2", code="y", language="python")
    manager.save(title="Js", code="z", language="javascript")

    langs = manager.list_languages()
    assert langs["python"] == 2
    assert langs["javascript"] == 1


def test_export(manager):
    manager.save(title="Export Me", code="x = 1", language="python")
    data = manager.export_all()
    assert len(data) == 1
    assert data[0]["title"] == "Export Me"


def test_usage_count_increments(manager):
    snippet = manager.save(title="Counter", code="x")
    assert snippet.usage_count == 0

    manager.get(snippet.id)
    manager.get(snippet.id)

    # Re-fetch to check count (get increments)
    retrieved = manager.get(snippet.id)
    assert retrieved.usage_count >= 2
