"""Module-level singletons for the observe system.

Initialised during gateway startup.
"""

from __future__ import annotations

from llmstack.observe.traces import TraceStore
from llmstack.observe.scoring import QualityScorer
from llmstack.observe.tracker import QualityTracker
from llmstack.observe.ab_testing import ABTestManager

_trace_store: TraceStore | None = None
_scorer: QualityScorer | None = None
_tracker: QualityTracker | None = None
_ab_manager: ABTestManager | None = None


def init_observe(
    trace_store: TraceStore | None = None,
    scorer: QualityScorer | None = None,
    tracker: QualityTracker | None = None,
    ab_manager: ABTestManager | None = None,
) -> None:
    global _trace_store, _scorer, _tracker, _ab_manager
    _trace_store = trace_store or TraceStore()
    _scorer = scorer or QualityScorer()
    _tracker = tracker or QualityTracker()
    _ab_manager = ab_manager or ABTestManager()


def get_trace_store() -> TraceStore | None:
    return _trace_store


def get_scorer() -> QualityScorer | None:
    return _scorer


def get_tracker() -> QualityTracker | None:
    return _tracker


def get_ab_manager() -> ABTestManager | None:
    return _ab_manager
