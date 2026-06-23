"""Tests for the benchmark egress proof (llmstack.benchmark.privacy)."""

from __future__ import annotations

import socket

from llmstack.benchmark.privacy import EgressProof, run_with_egress_proof


def test_no_connections_is_local_only() -> None:
    result, proof = run_with_egress_proof(lambda: 42)
    assert result == 42
    assert isinstance(proof, EgressProof)
    assert proof.is_local_only is True
    assert proof.total_connections == 0
    assert proof.external_connections == ()


def test_local_connection_recorded_but_still_local_only() -> None:
    def work():
        s = socket.socket()
        try:
            # Connecting to a closed local port still records the attempt.
            s.connect_ex(("127.0.0.1", 9))
        finally:
            s.close()
        return "done"

    result, proof = run_with_egress_proof(work)
    assert result == "done"
    assert proof.total_connections >= 1
    assert proof.is_local_only is True
    assert proof.external_connections == ()


def test_external_connection_flagged(monkeypatch) -> None:
    # Replace the OS connect with a no-op so no real packet ever leaves the test
    # host; the monitor still records the (external) target it was asked for,
    # which is all we need to exercise the flagging path. 8.8.8.8 is a public IP
    # that classifies as external.
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, addr: 0)
    external_ip = "8.8.8.8"

    def work():
        s = socket.socket()
        try:
            s.connect_ex((external_ip, 9))
        finally:
            s.close()

    _, proof = run_with_egress_proof(work)
    assert proof.is_local_only is False
    assert any(external_ip in c for c in proof.external_connections)


def test_proof_as_dict() -> None:
    _, proof = run_with_egress_proof(lambda: None)
    d = proof.as_dict()
    assert set(d) == {"is_local_only", "total_connections", "external_connections"}
