"""Reproducible benchmark suite for llmstack.

A provider-agnostic harness that measures local LLM latency/throughput, values
the run against dated cloud pricing, and proves zero external egress — producing
a deterministic, shareable report. See :mod:`llmstack.benchmark.spec` for the
task suite and :func:`run_benchmark` for the orchestrator.
"""

from __future__ import annotations
