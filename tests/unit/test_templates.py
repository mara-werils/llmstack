"""Tests for prompt template engine."""

import pytest
from llmstack.prompts.templates import TemplateManager, BUILTIN_TEMPLATES


@pytest.fixture
def manager(tmp_path):
    return TemplateManager(db_path=tmp_path / "test_templates.db")


def test_builtin_templates_exist(manager):
    templates = manager.list_all()
    assert len(templates) >= len(BUILTIN_TEMPLATES)
    names = {t.name for t in templates}
    assert "code-review" in names
    assert "unit-test" in names
    assert "debug" in names


def test_get_builtin(manager):
    template = manager.get("code-review")
    assert template is not None
    assert template.is_builtin
    assert "language" in template.variables


def test_render_with_variables(manager):
    rendered = manager.render("code-review", language="python", code="x = 1", focus="bugs")
    assert "python" in rendered
    assert "x = 1" in rendered
    assert "bugs" in rendered


def test_render_with_defaults(manager):
    rendered = manager.render("code-review", language="python", code="x = 1")
    # Should use default value for focus
    assert "python" in rendered
    assert "security, performance, readability" in rendered


def test_save_custom(manager):
    template = manager.save(
        name="my-template",
        template="Hello {{name}}, welcome to {{project}}!",
        description="A test template",
        category="test",
    )
    assert template.name == "my-template"
    assert "name" in template.variables
    assert "project" in template.variables


def test_render_custom(manager):
    manager.save(
        name="greet",
        template="Hello {{name}}!",
    )
    rendered = manager.render("greet", name="World")
    assert rendered == "Hello World!"


def test_delete_custom(manager):
    manager.save(name="temp", template="{{x}}")
    assert manager.delete("temp")
    assert manager.get("temp") is None


def test_cannot_delete_builtin(manager):
    assert not manager.delete("code-review")
    assert manager.get("code-review") is not None


def test_categories(manager):
    cats = manager.categories()
    assert "development" in cats
    assert "testing" in cats
    assert "documentation" in cats


def test_list_by_category(manager):
    templates = manager.list_all(category="testing")
    for t in templates:
        assert t.category == "testing"


def test_auto_detect_variables(manager):
    template = manager.save(
        name="auto-vars",
        template="{{foo}} and {{bar|default_val}} and {{baz}}",
    )
    assert set(template.variables) == {"foo", "bar", "baz"}


def test_render_missing_template(manager):
    with pytest.raises(ValueError, match="Template not found"):
        manager.render("nonexistent")
