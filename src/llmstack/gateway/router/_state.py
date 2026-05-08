"""Module-level singletons for the router and stats tracker.

Initialised by ``init_router()`` during gateway startup.
"""

from __future__ import annotations

from llmstack.gateway.router.router import ModelRouter
from llmstack.gateway.router.stats import RouterStats

_router: ModelRouter | None = None
_stats: RouterStats | None = None


def init_router(router: ModelRouter, stats: RouterStats) -> None:
    global _router, _stats
    _router = router
    _stats = stats


def get_router() -> ModelRouter | None:
    return _router


def get_stats() -> RouterStats | None:
    return _stats
