"""Dead code detector — find unused functions, classes, imports, and variables."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeadCodeItem:
    """A piece of potentially dead code."""
    type: str  # function, class, import, variable
    name: str
    file: str
    line: int
    confidence: str  # high, medium, low
    reason: str


class DeadCodeDetector:
    """Detects potentially unused code in Python projects."""

    IGNORE_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", "dist", "build", ".tox", ".eggs"}

    # Names that are typically used implicitly
    IMPLICIT_NAMES = {
        "__init__", "__main__", "__all__", "__version__",
        "__str__", "__repr__", "__eq__", "__hash__", "__len__",
        "__getitem__", "__setitem__", "__delitem__", "__contains__",
        "__iter__", "__next__", "__enter__", "__exit__",
        "__call__", "__bool__", "__int__", "__float__",
        "__add__", "__sub__", "__mul__", "__truediv__",
        "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
        "__post_init__", "__init_subclass__", "__class_getitem__",
        "setUp", "tearDown", "setUpClass", "tearDownClass",
        "main", "app", "cli", "setup",
    }

    def __init__(self, directory: Path):
        self.directory = directory
        self.definitions: list[dict] = []
        self.references: set[str] = set()
        self.all_exports: set[str] = set()

    def scan(self) -> list[DeadCodeItem]:
        """Scan project and return potentially dead code."""
        py_files = [
            p for p in sorted(self.directory.rglob("*.py"))
            if not any(part in self.IGNORE_DIRS for part in p.parts)
        ]

        # Pass 1: Collect all definitions and references
        for file_path in py_files:
            self._analyze_file(file_path)

        # Pass 2: Find unreferenced definitions
        dead_items = []
        for defn in self.definitions:
            name = defn["name"]
            short_name = name.split(".")[-1]

            # Skip implicit names
            if short_name in self.IMPLICIT_NAMES:
                continue
            # Skip private names starting with _ (often used internally)
            if short_name.startswith("_") and not short_name.startswith("__"):
                continue
            # Skip names in __all__
            if short_name in self.all_exports:
                continue
            # Skip test functions
            if short_name.startswith("test_"):
                continue

            # Check if name is referenced anywhere
            if short_name not in self.references:
                confidence = "high" if defn["type"] in ("function", "class") else "medium"
                # Lower confidence for short names (more likely to be referenced indirectly)
                if len(short_name) <= 3:
                    confidence = "low"

                dead_items.append(DeadCodeItem(
                    type=defn["type"],
                    name=name,
                    file=defn["file"],
                    line=defn["line"],
                    confidence=confidence,
                    reason=f"No references found to '{short_name}' in the project",
                ))

        # Detect unused imports
        dead_items.extend(self._find_unused_imports(py_files))

        return sorted(dead_items, key=lambda x: (
            {"high": 0, "medium": 1, "low": 2}.get(x.confidence, 3),
            x.type, x.file, x.line,
        ))

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for definitions and references."""
        try:
            content = file_path.read_text(errors="replace")
            tree = ast.parse(content)
        except (SyntaxError, OSError):
            return

        rel_path = str(file_path)

        # Collect definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.definitions.append({
                    "type": "function",
                    "name": node.name,
                    "file": rel_path,
                    "line": node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                self.definitions.append({
                    "type": "class",
                    "name": node.name,
                    "file": rel_path,
                    "line": node.lineno,
                })

        # Collect __all__ exports
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    self.all_exports.add(elt.value)

        # Collect references (all Name nodes that aren't definitions)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.references.add(node.id)
            elif isinstance(node, ast.Attribute):
                self.references.add(node.attr)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.references.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    self.references.add(node.func.attr)

        # Also scan string references (for dynamic usage)
        for match in re.finditer(r'["\'](\w+)["\']', content):
            self.references.add(match.group(1))

    def _find_unused_imports(self, py_files: list[Path]) -> list[DeadCodeItem]:
        """Find imports that are never used in the file."""
        items = []

        for file_path in py_files:
            try:
                content = file_path.read_text(errors="replace")
                tree = ast.parse(content)
            except (SyntaxError, OSError):
                continue

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[-1]
                        imports.append((name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.names:
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            name = alias.asname or alias.name
                            imports.append((name, node.lineno))

            # Check if import is used in the file
            for imp_name, imp_line in imports:
                if imp_name.startswith("_"):
                    continue

                # Count occurrences (excluding the import line itself)
                lines = content.split("\n")
                used = False
                for i, line in enumerate(lines, 1):
                    if i == imp_line:
                        continue
                    if re.search(rf'\b{re.escape(imp_name)}\b', line):
                        used = True
                        break

                if not used:
                    items.append(DeadCodeItem(
                        type="import",
                        name=imp_name,
                        file=str(file_path),
                        line=imp_line,
                        confidence="high",
                        reason=f"Import '{imp_name}' is never used in this file",
                    ))

        return items
