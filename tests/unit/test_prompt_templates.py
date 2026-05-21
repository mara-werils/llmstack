"""Tests for the prompt template management system."""

import pytest

from llmstack.gateway.prompt_templates import (
    TemplateStore,
    PromptTemplate,
    _extract_variables,
    BUILTIN_TEMPLATES,
)


class TestExtractVariables:
    def test_single_variable(self):
        assert _extract_variables("Hello {{name}}!") == ["name"]

    def test_multiple_variables(self):
        result = _extract_variables("{{greeting}} {{name}}, welcome to {{place}}")
        assert result == ["greeting", "name", "place"]

    def test_no_variables(self):
        assert _extract_variables("Hello world!") == []

    def test_duplicate_variables(self):
        assert _extract_variables("{{x}} and {{x}} again") == ["x"]

    def test_nested_braces_ignored(self):
        assert _extract_variables("{{{notavar}}}") == ["notavar"]


class TestTemplateStore:
    def setup_method(self):
        self.store = TemplateStore()

    def test_create_template(self):
        t = self.store.create(
            name="test", content="Hello {{name}}!", description="A test"
        )
        assert t.name == "test"
        assert t.current_version == 1
        assert len(t.versions) == 1
        assert t.versions[0].variables == ["name"]

    def test_create_duplicate_raises(self):
        self.store.create(name="dup", content="foo")
        with pytest.raises(ValueError, match="already exists"):
            self.store.create(name="dup", content="bar")

    def test_get_by_name(self):
        self.store.create(name="lookup", content="test")
        result = self.store.get("lookup")
        assert result is not None
        assert result.name == "lookup"

    def test_get_by_id(self):
        t = self.store.create(name="byid", content="test")
        result = self.store.get(t.id)
        assert result is not None
        assert result.id == t.id

    def test_get_nonexistent(self):
        assert self.store.get("nonexistent") is None

    def test_update_creates_new_version(self):
        self.store.create(name="versioned", content="v1")
        updated = self.store.update("versioned", content="v2")
        assert updated is not None
        assert updated.current_version == 2
        assert len(updated.versions) == 2
        assert updated.get_current().content == "v2"

    def test_delete(self):
        self.store.create(name="deleteme", content="bye")
        assert self.store.delete("deleteme") is True
        assert self.store.get("deleteme") is None

    def test_delete_nonexistent(self):
        assert self.store.delete("nope") is False

    def test_render_with_variables(self):
        self.store.create(name="greet", content="Hello {{name}}, you are {{role}}!")
        result = self.store.render("greet", {"name": "Alice", "role": "admin"})
        assert result == "Hello Alice, you are admin!"

    def test_render_missing_template(self):
        assert self.store.render("missing") is None

    def test_list_all(self):
        self.store.create(name="a", content="aa", category="dev")
        self.store.create(name="b", content="bb", category="ops")
        self.store.create(name="c", content="cc", category="dev")

        all_templates = self.store.list_all()
        assert len(all_templates) == 3

        dev_only = self.store.list_all(category="dev")
        assert len(dev_only) == 2

    def test_list_by_tag(self):
        self.store.create(name="tagged", content="x", tags=["python"])
        self.store.create(name="untagged", content="y", tags=["go"])
        results = self.store.list_all(tag="python")
        assert len(results) == 1
        assert results[0].name == "tagged"

    def test_rollback(self):
        self.store.create(name="rb", content="v1")
        self.store.update("rb", content="v2")
        self.store.update("rb", content="v3")

        rolled = self.store.rollback("rb", version=1)
        assert rolled is not None
        assert rolled.current_version == 1
        assert rolled.get_current().content == "v1"

    def test_rollback_invalid_version(self):
        self.store.create(name="rb2", content="v1")
        with pytest.raises(ValueError, match="not found"):
            self.store.rollback("rb2", version=99)

    def test_search(self):
        self.store.create(name="python-helper", content="x", description="helps with python")
        self.store.create(name="go-helper", content="y", description="helps with go")
        results = self.store.search("python")
        assert len(results) == 1
        assert results[0].name == "python-helper"

    def test_count(self):
        assert self.store.count == 0
        self.store.create(name="one", content="1")
        assert self.store.count == 1

    def test_to_dict(self):
        t = self.store.create(name="dict_test", content="Hello {{x}}!")
        d = t.to_dict()
        assert d["name"] == "dict_test"
        assert d["variables"] == ["x"]
        assert d["current_version"] == 1


class TestBuiltinTemplates:
    def test_builtin_templates_are_valid(self):
        store = TemplateStore()
        for bt in BUILTIN_TEMPLATES:
            t = store.create(**bt)
            assert t.name == bt["name"]
            current = t.get_current()
            assert current is not None
            assert len(current.variables) > 0

    def test_builtin_count(self):
        assert len(BUILTIN_TEMPLATES) >= 5
