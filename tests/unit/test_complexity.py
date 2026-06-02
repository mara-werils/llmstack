"""Tests for code complexity analyzer."""

import tempfile
from pathlib import Path

import pytest

from llmstack.analyze.complexity import analyze_python_file, analyze_directory


@pytest.fixture
def simple_py(tmp_path):
    code = '''
def hello():
    print("hello")

def add(a, b):
    return a + b
'''
    f = tmp_path / "simple.py"
    f.write_text(code)
    return f


@pytest.fixture
def complex_py(tmp_path):
    code = '''
def complex_function(data, config, mode):
    if mode == "a":
        for item in data:
            if item > 0:
                if config.get("verbose"):
                    print(item)
                elif config.get("debug"):
                    print(f"debug: {item}")
                else:
                    pass
            elif item < 0:
                try:
                    process(item)
                except ValueError:
                    handle_error(item)
                except TypeError:
                    handle_type_error(item)
    elif mode == "b":
        while True:
            if not data:
                break
            item = data.pop()
            if item and item > 0 or item == -1:
                yield item
    else:
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if data[i] > data[j]:
                    data[i], data[j] = data[j], data[i]
    return data
'''
    f = tmp_path / "complex.py"
    f.write_text(code)
    return f


def test_simple_file_analysis(simple_py):
    metrics = analyze_python_file(simple_py)
    assert metrics is not None
    assert metrics.functions == 2
    assert metrics.code_lines > 0
    assert len(metrics.function_metrics) == 2

    # Simple functions should have low complexity
    for fm in metrics.function_metrics:
        assert fm.cyclomatic <= 3
        assert fm.grade in ("A", "B")


def test_complex_file_analysis(complex_py):
    metrics = analyze_python_file(complex_py)
    assert metrics is not None
    assert metrics.functions >= 1

    # Complex function should have high complexity
    complex_func = metrics.function_metrics[0]
    assert complex_func.cyclomatic > 5
    assert complex_func.nested_depth >= 2


def test_maintainability_index(simple_py):
    metrics = analyze_python_file(simple_py)
    assert metrics is not None
    # Simple code should have high maintainability
    assert metrics.maintainability_index > 50
    assert metrics.grade in ("A", "B", "C")


def test_grade_assignment(simple_py, complex_py):
    simple_metrics = analyze_python_file(simple_py)
    complex_metrics = analyze_python_file(complex_py)

    assert simple_metrics is not None
    assert complex_metrics is not None

    # Simple should grade better than complex
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    for sm in simple_metrics.function_metrics:
        assert grade_order[sm.grade] <= 2  # A, B, or C


def test_analyze_directory(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): return 1\n")
    (tmp_path / "b.py").write_text("def bar(x):\n  if x: return 1\n  return 0\n")

    results = analyze_directory(tmp_path)
    assert len(results) == 2


def test_syntax_error_handling(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def foo(:\n  pass\n")
    result = analyze_python_file(bad)
    assert result is None


def test_empty_file(tmp_path):
    empty = tmp_path / "empty.py"
    empty.write_text("")
    result = analyze_python_file(empty)
    assert result is not None
    assert result.functions == 0


def test_class_methods(tmp_path):
    code = '''
class MyClass:
    def method_a(self):
        return 1

    def method_b(self, x):
        if x > 0:
            return x
        return -x
'''
    f = tmp_path / "cls.py"
    f.write_text(code)
    metrics = analyze_python_file(f)
    assert metrics is not None
    assert metrics.classes == 1
    assert metrics.functions == 2
