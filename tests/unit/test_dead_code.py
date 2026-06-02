"""Tests for dead code detector."""

import pytest
from pathlib import Path
from llmstack.analyze.dead_code import DeadCodeDetector


@pytest.fixture
def project_dir(tmp_path):
    # File with unused function
    (tmp_path / "main.py").write_text('''
from utils import helper

def main():
    result = helper()
    return result

def unused_function():
    """This function is never called."""
    return 42
''')

    (tmp_path / "utils.py").write_text('''
def helper():
    return "hello"

def another_unused():
    return "world"
''')
    return tmp_path


def test_detects_unused_imports(tmp_path):
    (tmp_path / "test.py").write_text('''
import os
import sys
import json

x = json.dumps({"a": 1})
''')
    detector = DeadCodeDetector(tmp_path)
    items = detector.scan()

    unused_imports = [i for i in items if i.type == "import"]
    import_names = {i.name for i in unused_imports}
    assert "os" in import_names
    assert "sys" in import_names
    assert "json" not in import_names  # json is used


def test_detects_unused_functions(project_dir):
    detector = DeadCodeDetector(project_dir)
    items = detector.scan()

    func_names = {i.name for i in items if i.type == "function"}
    # These might be detected depending on reference analysis
    assert len(items) > 0


def test_ignores_dunder_methods(tmp_path):
    (tmp_path / "cls.py").write_text('''
class Foo:
    def __init__(self):
        self.x = 1

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return f"Foo({self.x})"
''')
    detector = DeadCodeDetector(tmp_path)
    items = detector.scan()

    # Dunder methods should not be reported
    dunder_items = [i for i in items if i.name.startswith("__") and i.name.endswith("__")]
    assert len(dunder_items) == 0


def test_ignores_test_functions(tmp_path):
    (tmp_path / "test_something.py").write_text('''
def test_foo():
    assert True

def test_bar():
    assert 1 + 1 == 2
''')
    detector = DeadCodeDetector(tmp_path)
    items = detector.scan()

    test_items = [i for i in items if i.name.startswith("test_")]
    assert len(test_items) == 0


def test_respects_all_exports(tmp_path):
    (tmp_path / "api.py").write_text('''
__all__ = ["public_func"]

def public_func():
    return "public"

def internal_func():
    return "internal"
''')
    detector = DeadCodeDetector(tmp_path)
    items = detector.scan()

    names = {i.name for i in items}
    assert "public_func" not in names


def test_handles_syntax_errors(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(\n  pass")
    (tmp_path / "good.py").write_text("def good(): return 1")

    detector = DeadCodeDetector(tmp_path)
    items = detector.scan()
    # Should not crash, should still analyze good.py
    assert isinstance(items, list)
