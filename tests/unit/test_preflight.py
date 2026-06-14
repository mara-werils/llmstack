"""Tests for stack pre-flight checks (port availability + required-port derivation)."""

from __future__ import annotations

import socket
from types import SimpleNamespace

from llmstack.core.preflight import (
    PortCheck,
    check_ports,
    is_port_available,
    port_owner,
    required_ports,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_is_port_available_toggles_with_a_live_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert is_port_available(port) is False  # held by our listener
    # Socket closed → port is free again.
    assert is_port_available(port) is True


def test_check_ports_reports_owner_only_when_taken() -> None:
    free = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]

        results = check_ports([(free, "Free"), (taken, "Taken")])

    by_service = {r.service: r for r in results}
    assert by_service["Free"].available is True
    assert by_service["Free"].owner is None
    assert by_service["Taken"].available is False
    assert isinstance(by_service["Taken"], PortCheck)


def test_port_owner_returns_none_for_free_port() -> None:
    assert port_owner(_free_port()) is None


def test_required_ports_derives_from_config_and_includes_observe_when_enabled() -> None:
    config = SimpleNamespace(
        services=SimpleNamespace(
            vectors=SimpleNamespace(port=6333),
            cache=SimpleNamespace(port=6379),
        ),
        gateway=SimpleNamespace(port=8000),
        observe=SimpleNamespace(metrics=True, dashboard_port=8080),
    )
    ports = required_ports(config)
    services = {svc for _, svc in ports}
    assert {"Qdrant", "Redis", "Gateway", "Prometheus", "Grafana"} <= services
    # Inference (Ollama 11434) is intentionally excluded.
    assert 11434 not in {p for p, _ in ports}


def test_required_ports_omits_observe_when_metrics_disabled() -> None:
    config = SimpleNamespace(
        services=SimpleNamespace(
            vectors=SimpleNamespace(port=6333),
            cache=SimpleNamespace(port=6379),
        ),
        gateway=SimpleNamespace(port=8000),
        observe=SimpleNamespace(metrics=False, dashboard_port=8080),
    )
    services = {svc for _, svc in required_ports(config)}
    assert services == {"Qdrant", "Redis", "Gateway"}
