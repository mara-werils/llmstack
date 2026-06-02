"""Code complexity analyzer — cyclomatic complexity, cognitive complexity, and maintainability index."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from math import log2


@dataclass
class FunctionMetrics:
    """Metrics for a single function/method."""
    name: str
    file: str
    line: int
    end_line: int
    cyclomatic: int
    cognitive: int
    lines_of_code: int
    parameters: int
    returns: int
    nested_depth: int
    grade: str  # A, B, C, D, F


@dataclass
class FileMetrics:
    """Metrics for a file."""
    file: str
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    avg_complexity: float
    max_complexity: int
    maintainability_index: float
    grade: str
    function_metrics: list[FunctionMetrics]


def _grade_complexity(complexity: int) -> str:
    """Grade cyclomatic complexity."""
    if complexity <= 5:
        return "A"
    elif complexity <= 10:
        return "B"
    elif complexity <= 20:
        return "C"
    elif complexity <= 30:
        return "D"
    return "F"


def _grade_maintainability(mi: float) -> str:
    """Grade maintainability index."""
    if mi >= 80:
        return "A"
    elif mi >= 60:
        return "B"
    elif mi >= 40:
        return "C"
    elif mi >= 20:
        return "D"
    return "F"


def _cyclomatic_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity

    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1

    return complexity


def _cognitive_complexity(node: ast.AST, depth: int = 0) -> int:
    """Calculate cognitive complexity (how hard to understand)."""
    total = 0

    for child in ast.iter_child_nodes(node):
        increment = 0

        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            increment = 1 + depth
        elif isinstance(child, ast.ExceptHandler):
            increment = 1 + depth
        elif isinstance(child, ast.BoolOp):
            increment = 1
        elif isinstance(child, (ast.Break, ast.Continue)):
            increment = 1

        total += increment

        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
            total += _cognitive_complexity(child, depth + 1)
        else:
            total += _cognitive_complexity(child, depth)

    return total


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Calculate maximum nesting depth."""
    max_depth = depth

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                              ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler)):
            max_depth = max(max_depth, _max_nesting(child, depth + 1))
        else:
            max_depth = max(max_depth, _max_nesting(child, depth))

    return max_depth


def _count_returns(node: ast.AST) -> int:
    """Count return statements."""
    return sum(1 for child in ast.walk(node) if isinstance(child, ast.Return))


def analyze_python_file(file_path: Path) -> FileMetrics | None:
    """Analyze a Python file for complexity metrics."""
    try:
        content = file_path.read_text(errors="replace")
        tree = ast.parse(content)
    except (SyntaxError, OSError):
        return None

    lines = content.split("\n")
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    code_lines = total_lines - blank_lines - comment_lines

    function_metrics = []
    class_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_count += 1

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _cyclomatic_complexity(node)
            cog = _cognitive_complexity(node)
            end_line = getattr(node, 'end_lineno', node.lineno)
            loc = end_line - node.lineno + 1
            params = len(node.args.args) + len(node.args.kwonlyargs)
            returns = _count_returns(node)
            depth = _max_nesting(node)

            # Class method prefix
            name = node.name
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for item in parent.body:
                        if item is node:
                            name = f"{parent.name}.{node.name}"
                            break

            function_metrics.append(FunctionMetrics(
                name=name,
                file=str(file_path),
                line=node.lineno,
                end_line=end_line,
                cyclomatic=cc,
                cognitive=cog,
                lines_of_code=loc,
                parameters=params,
                returns=returns,
                nested_depth=depth,
                grade=_grade_complexity(cc),
            ))

    # Calculate maintainability index
    # MI = max(0, (171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)) * 100 / 171)
    avg_cc = sum(f.cyclomatic for f in function_metrics) / max(1, len(function_metrics))
    volume = max(1, code_lines * log2(max(1, len(set(content.split())))))
    mi = max(0, (171 - 5.2 * log2(max(1, volume)) - 0.23 * avg_cc - 16.2 * log2(max(1, code_lines))) * 100 / 171)

    max_cc = max((f.cyclomatic for f in function_metrics), default=0)

    return FileMetrics(
        file=str(file_path),
        total_lines=total_lines,
        code_lines=code_lines,
        comment_lines=comment_lines,
        blank_lines=blank_lines,
        functions=len(function_metrics),
        classes=class_count,
        avg_complexity=round(avg_cc, 1),
        max_complexity=max_cc,
        maintainability_index=round(mi, 1),
        grade=_grade_maintainability(mi),
        function_metrics=function_metrics,
    )


def analyze_directory(directory: Path, threshold: int = 10) -> list[FileMetrics]:
    """Analyze all Python files in a directory."""
    ignore = {"__pycache__", ".git", "node_modules", "venv", ".venv", "dist", "build", ".tox"}
    results = []

    for p in sorted(directory.rglob("*.py")):
        if any(part in ignore for part in p.parts):
            continue
        metrics = analyze_python_file(p)
        if metrics:
            results.append(metrics)

    return results
