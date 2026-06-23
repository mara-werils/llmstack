"""The benchmark task suite — fixed, versioned, and deterministic.

The whole point of these tasks is reproducibility: the prompts never change for a
given suite version, so two people running the same suite are measuring the same
work. Bump :attr:`BenchmarkSuite.version` whenever a suite's tasks change.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

# Task categories used to group results in the report.
CATEGORIES = ("latency", "coding", "reasoning")


@dataclass(frozen=True)
class BenchmarkTask:
    """A single prompt to send to the model under test."""

    id: str
    category: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkSuite:
    """A named, versioned collection of benchmark tasks."""

    name: str
    version: str
    tasks: tuple[BenchmarkTask, ...]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[BenchmarkTask]:
        return iter(self.tasks)

    def categories(self) -> list[str]:
        """Distinct categories present, in first-seen order."""
        seen: list[str] = []
        for task in self.tasks:
            if task.category not in seen:
                seen.append(task.category)
        return seen

    def filter(self, category: str) -> BenchmarkSuite:
        """Return a sub-suite containing only tasks in ``category``."""
        tasks = tuple(t for t in self.tasks if t.category == category)
        return BenchmarkSuite(name=f"{self.name}:{category}", version=self.version, tasks=tasks)


_DEFAULT_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask("latency-1", "latency", "Reply with the single word: ok"),
    BenchmarkTask("latency-2", "latency", "What is 2 + 2? Answer with just the number."),
    BenchmarkTask("coding-1", "coding", "Write a Python function that reverses a string."),
    BenchmarkTask(
        "coding-2",
        "coding",
        "Explain what this code does in one sentence:\n"
        "def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)",
    ),
    BenchmarkTask(
        "reasoning-1",
        "reasoning",
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the "
        "ball. How much does the ball cost? Answer with just the amount.",
    ),
    BenchmarkTask(
        "reasoning-2",
        "reasoning",
        "List three trade-offs of microservices versus a monolith, briefly.",
    ),
)


DEFAULT_SUITE = BenchmarkSuite(name="default", version="1", tasks=_DEFAULT_TASKS)

SUITES: dict[str, BenchmarkSuite] = {DEFAULT_SUITE.name: DEFAULT_SUITE}


def get_suite(name: str = "default") -> BenchmarkSuite:
    """Return a built-in suite by name, or raise ``KeyError``."""
    return SUITES[name]
