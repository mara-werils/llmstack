"""Tests for stack pre-flight checks (port availability + required-port derivation)."""

from __future__ import annotations

import socket
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llmstack.core.preflight import (
    PortCheck,
    check_ports,
    docker_status,
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


def _fake_conn(port, status, pid=1234):
    return SimpleNamespace(laddr=SimpleNamespace(port=port), status=status, pid=pid)


def _fake_psutil(connections, process_name_side_effect=None):
    fake = MagicMock()
    fake.CONN_LISTEN = "LISTEN"
    fake.net_connections.return_value = connections
    if process_name_side_effect is not None:
        fake.Process.side_effect = process_name_side_effect
    else:
        fake.Process.return_value = MagicMock(name=lambda: "myproc")
    return fake


def test_port_owner_not_installed():
    with patch.dict(sys.modules, {"psutil": None}):
        assert port_owner(1234) is None


def test_port_owner_finds_listening_process():
    fake = _fake_psutil([_fake_conn(1234, "LISTEN", pid=42)])
    fake.Process.return_value.name.return_value = "myproc"
    with patch.dict(sys.modules, {"psutil": fake}):
        assert port_owner(1234) == "myproc"


def test_port_owner_listening_without_pid_returns_none():
    fake = _fake_psutil([_fake_conn(1234, "LISTEN", pid=0)])
    with patch.dict(sys.modules, {"psutil": fake}):
        assert port_owner(1234) is None


def test_port_owner_process_lookup_fails_returns_none():
    fake = _fake_psutil([_fake_conn(1234, "LISTEN", pid=42)])
    fake.Process.side_effect = Exception("no such process")
    with patch.dict(sys.modules, {"psutil": fake}):
        assert port_owner(1234) is None


def test_port_owner_no_matching_connection_returns_none():
    fake = _fake_psutil([_fake_conn(9999, "LISTEN")])
    with patch.dict(sys.modules, {"psutil": fake}):
        assert port_owner(1234) is None


def test_port_owner_net_connections_raises_returns_none():
    fake = MagicMock()
    fake.net_connections.side_effect = PermissionError("denied")
    with patch.dict(sys.modules, {"psutil": fake}):
        assert port_owner(1234) is None


def test_docker_status_not_installed():
    with patch("llmstack.core.preflight.shutil.which", return_value=None):
        msg = docker_status()
    assert "not installed" in msg


def test_docker_status_daemon_unreachable():
    fake_docker = MagicMock()
    fake_docker.from_env.side_effect = RuntimeError("daemon down")
    with (
        patch("llmstack.core.preflight.shutil.which", return_value="/usr/bin/docker"),
        patch.dict(sys.modules, {"docker": fake_docker}),
    ):
        msg = docker_status()
    assert "daemon isn't reachable" in msg


def test_docker_status_healthy_returns_none():
    fake_client = MagicMock()
    fake_docker = MagicMock()
    fake_docker.from_env.return_value = fake_client
    with (
        patch("llmstack.core.preflight.shutil.which", return_value="/usr/bin/docker"),
        patch.dict(sys.modules, {"docker": fake_docker}),
    ):
        assert docker_status() is None
    fake_client.ping.assert_called_once()
