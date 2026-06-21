"""Tests for the runtime egress monitor (llmstack.core.egress)."""

from __future__ import annotations

import socket
import threading

import pytest

from llmstack.core.egress import (
    Connection,
    EgressMonitor,
    ExternalEgressError,
    _split_address,
    assert_local_only,
    is_local_host,
    monitor_egress,
)


@pytest.mark.parametrize(
    "host",
    [
        "",
        "localhost",
        "host.docker.internal",
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.5",
        "172.16.9.9",
        "169.254.1.1",  # link-local
        "0.0.0.0",  # unspecified
        "::1",
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 unique-local (private)
        "foo.local",
        "llmstack-ollama",
    ],
)
def test_local_hosts(host: str) -> None:
    assert is_local_host(host) is True


@pytest.mark.parametrize(
    "host",
    ["8.8.8.8", "1.1.1.1", "9.9.9.9", "api.openai.com", "llmstack.evil.com", "2606:4700::1"],
)
def test_external_hosts(host: str) -> None:
    assert is_local_host(host) is False


def test_connection_properties() -> None:
    local = Connection("127.0.0.1", 8000)
    external = Connection("8.8.8.8", 443)
    assert local.is_local and not external.is_local
    assert str(local) == "127.0.0.1:8000"


def test_split_address_variants() -> None:
    assert _split_address(("127.0.0.1", 80)) == ("127.0.0.1", 80)
    assert _split_address(("::1", 80, 0, 0)) == ("::1", 80)  # IPv6 4-tuple
    assert _split_address("/tmp/sock.unix") == (None, 0)  # AF_UNIX path
    assert _split_address(None) == (None, 0)


def test_monitor_external_record_without_io() -> None:
    mon = EgressMonitor()
    mon.record(("8.8.8.8", 443))
    mon.record(("127.0.0.1", 8000))
    mon.record("/tmp/ipc")  # ignored — local IPC
    assert len(mon.connections) == 2
    assert not mon.is_local_only
    assert [str(c) for c in mon.external] == ["8.8.8.8:443"]


def test_assert_local_only_raises_on_external() -> None:
    mon = EgressMonitor()
    mon.record(("9.9.9.9", 443))
    with pytest.raises(ExternalEgressError, match="9.9.9.9:443"):
        assert_local_only(mon)


def test_assert_local_only_passes_when_clean() -> None:
    mon = EgressMonitor()
    mon.record(("127.0.0.1", 8000))
    assert_local_only(mon)  # no raise
    assert mon.is_local_only


def test_monitor_records_real_local_connect() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    accepted: list[socket.socket] = []

    def accept() -> None:
        conn, _ = server.accept()
        accepted.append(conn)

    thread = threading.Thread(target=accept)
    thread.start()

    with monitor_egress() as mon:
        client = socket.create_connection(("127.0.0.1", port))
        client.close()

    thread.join(timeout=2)
    for conn in accepted:
        conn.close()
    server.close()

    assert any(c.port == port and c.is_local for c in mon.connections)
    assert mon.is_local_only


def test_monitor_restores_socket_methods() -> None:
    before_connect = socket.socket.connect
    before_connect_ex = socket.socket.connect_ex
    with monitor_egress():
        assert socket.socket.connect is not before_connect
    assert socket.socket.connect is before_connect
    assert socket.socket.connect_ex is before_connect_ex


def test_monitor_records_connect_ex() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    with monitor_egress() as mon:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect_ex(("127.0.0.1", port))
        client.close()

    server.close()
    assert any(c.port == port for c in mon.connections)
