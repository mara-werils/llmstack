"""Runtime egress monitor — prove a code path makes no external network calls.

This complements the static :func:`llmstack.core.privacy.audit_privacy` check:
where the audit inspects *configuration*, this monitor observes the *actual*
outbound socket connections a block of code makes and flags any that leave the
local machine or private network. It powers a reproducible "no external egress"
proof that can run in CI.

Patching is scoped to a context manager and fully restored on exit, so it is
safe to use around any workload.
"""

from __future__ import annotations

import ipaddress
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from llmstack.core.privacy import is_local_url

# Hostnames (not IP literals) that always resolve to this machine.
_LOCAL_HOSTNAMES = {"localhost", "host.docker.internal"}


def is_local_host(host: str) -> bool:
    """Return True if ``host`` (hostname or IP literal) is local/private.

    IP literals are classified with :mod:`ipaddress` (loopback, private,
    link-local, unspecified). Non-IP hostnames fall back to the same rules as
    the static privacy audit (``localhost``, ``*.local``, dotless ``llmstack*``).
    """
    if not host:
        return True
    candidate = host.strip().lower()
    if candidate in _LOCAL_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return is_local_url(candidate)
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified


@dataclass
class Connection:
    """A single outbound connection attempt."""

    host: str
    port: int

    @property
    def is_local(self) -> bool:
        return is_local_host(self.host)

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


def _split_address(address: object) -> tuple[str | None, int]:
    """Extract ``(host, port)`` from a socket address, or ``(None, 0)`` for IPC."""
    if isinstance(address, tuple) and len(address) >= 2:
        return str(address[0]), int(address[1])
    # AF_UNIX (a path) or anything else: local IPC, never network egress.
    return None, 0


@dataclass
class EgressMonitor:
    """Records outbound socket connections and flags those that leave the host."""

    connections: list[Connection] = field(default_factory=list)

    @property
    def external(self) -> list[Connection]:
        return [c for c in self.connections if not c.is_local]

    @property
    def is_local_only(self) -> bool:
        return not self.external

    def record(self, address: object) -> None:
        host, port = _split_address(address)
        if host is not None:
            self.connections.append(Connection(host, port))


class ExternalEgressError(RuntimeError):
    """Raised when external network egress is detected where none is allowed."""


@contextmanager
def monitor_egress() -> Iterator[EgressMonitor]:
    """Yield an :class:`EgressMonitor` recording every connect during the block."""
    mon = EgressMonitor()
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def patched_connect(self: socket.socket, address: object, *args: object) -> object:
        mon.record(address)
        return real_connect(self, address, *args)  # type: ignore[arg-type]

    def patched_connect_ex(self: socket.socket, address: object, *args: object) -> object:
        mon.record(address)
        return real_connect_ex(self, address, *args)  # type: ignore[arg-type]

    socket.socket.connect = patched_connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = patched_connect_ex  # type: ignore[method-assign,assignment]
    try:
        yield mon
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]


def assert_local_only(mon: EgressMonitor) -> None:
    """Raise :class:`ExternalEgressError` if the monitor saw external egress."""
    if not mon.is_local_only:
        targets = ", ".join(str(c) for c in mon.external)
        raise ExternalEgressError(f"External network egress detected: {targets}")
