"""Guarantee test: core offline code paths make zero outbound connections.

This is the runtime half of LLMStack's privacy promise. If any future change
introduces telemetry or a phone-home on these paths, this test fails loudly.
"""

from __future__ import annotations

from llmstack.config.schema import StackConfig
from llmstack.core.egress import monitor_egress
from llmstack.core.privacy import audit_privacy, is_local_url


def test_default_config_and_audit_make_no_connections() -> None:
    with monitor_egress() as mon:
        config = StackConfig()
        report = audit_privacy(config)
        _ = report.verdict
        _ = is_local_url("http://localhost:8000")
    assert mon.connections == []
    assert mon.is_local_only


def test_privacy_audit_is_local_only_under_monitor() -> None:
    config = StackConfig()
    with monitor_egress() as mon:
        for _ in range(5):
            audit_privacy(config)
    assert mon.is_local_only
