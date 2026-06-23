"""Tests for the benchmark task suite (llmstack.benchmark.spec)."""

from __future__ import annotations

import pytest

from llmstack.benchmark.spec import (
    CATEGORIES,
    DEFAULT_SUITE,
    SUITES,
    BenchmarkSuite,
    BenchmarkTask,
    get_suite,
)


def test_default_suite_is_non_empty_and_versioned() -> None:
    assert len(DEFAULT_SUITE) > 0
    assert DEFAULT_SUITE.version
    assert DEFAULT_SUITE.name == "default"


def test_task_ids_are_unique() -> None:
    ids = [t.id for t in DEFAULT_SUITE]
    assert len(ids) == len(set(ids))


def test_every_task_category_is_known() -> None:
    for task in DEFAULT_SUITE:
        assert task.category in CATEGORIES
        assert task.prompt.strip()


def test_categories_first_seen_order() -> None:
    cats = DEFAULT_SUITE.categories()
    assert cats == list(dict.fromkeys(t.category for t in DEFAULT_SUITE))


def test_filter_returns_only_matching_tasks() -> None:
    sub = DEFAULT_SUITE.filter("coding")
    assert len(sub) > 0
    assert all(t.category == "coding" for t in sub)
    assert sub.version == DEFAULT_SUITE.version
    assert sub.name.endswith(":coding")


def test_filter_unknown_category_is_empty() -> None:
    assert len(DEFAULT_SUITE.filter("nope")) == 0


def test_iter_and_len_consistent() -> None:
    assert len(list(iter(DEFAULT_SUITE))) == len(DEFAULT_SUITE)


def test_get_suite_default_and_unknown() -> None:
    assert get_suite() is DEFAULT_SUITE
    assert get_suite("default") is SUITES["default"]
    with pytest.raises(KeyError):
        get_suite("does-not-exist")


def test_suite_is_immutable() -> None:
    task = BenchmarkTask("x", "latency", "p")
    suite = BenchmarkSuite("s", "1", (task,))
    with pytest.raises(Exception):
        suite.version = "2"  # type: ignore[misc]
