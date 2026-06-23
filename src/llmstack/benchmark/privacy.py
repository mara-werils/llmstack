"""Attach a runtime "zero external egress" proof to a benchmark run.

This reuses :func:`llmstack.core.egress.monitor_egress` to observe every outbound
socket connection made while the benchmark executes, and summarises whether any
of them left the machine. A clean proof is the privacy half of the value story:
the local run was not just cheaper, it provably sent nothing off-device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TypeVar

from llmstack.core.egress import monitor_egress

T = TypeVar("T")


@dataclass(frozen=True)
class EgressProof:
    """Summary of the network egress observed during a measured block."""

    is_local_only: bool
    total_connections: int
    external_connections: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_with_egress_proof(fn: Callable[[], T]) -> tuple[T, EgressProof]:
    """Run ``fn`` under the egress monitor and return ``(result, proof)``."""
    with monitor_egress() as mon:
        result = fn()
    proof = EgressProof(
        is_local_only=mon.is_local_only,
        total_connections=len(mon.connections),
        external_connections=tuple(str(c) for c in mon.external),
    )
    return result, proof
