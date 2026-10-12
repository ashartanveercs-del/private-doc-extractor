"""Suite-wide guard: the tests never touch the internet.

Every test runs with outbound connections limited to loopback (a few tests start a local HTTP
server) and with DNS limited to literal addresses and "localhost". A test that accidentally
reaches for a real API fails loudly instead of passing on a networked laptop and failing in CI.

`no_network_at_all` is stricter, for the local-tier tests: it also refuses loopback and records
every attempt, so "nothing leaves the machine" is asserted, not assumed.
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address):
    if not _is_loopback(address):
        raise OSError(f"network access is disabled in tests (tried {address!r})")
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_loopback(address):
        raise OSError(f"network access is disabled in tests (tried {address!r})")
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host in (None, "localhost"):
        return _real_getaddrinfo(host, *args, **kwargs)
    try:
        ipaddress.ip_address(str(host).split("%")[0])
    except ValueError:
        raise socket.gaierror(f"DNS is disabled in tests (tried {host!r})") from None
    return _real_getaddrinfo(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)


class NetworkTripwire:
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def trip(self, what: str):
        self.attempts.append(what)
        raise OSError(f"tripwire: {what}")


@pytest.fixture
def no_network_at_all(monkeypatch) -> NetworkTripwire:
    wire = NetworkTripwire()
    monkeypatch.setattr(socket.socket, "connect", lambda self, a: wire.trip(f"connect {a!r}"))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, a: wire.trip(f"connect {a!r}"))
    monkeypatch.setattr(socket, "create_connection",
                        lambda a, *x, **k: wire.trip(f"create_connection {a!r}"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, *x, **k: wire.trip(f"getaddrinfo {h!r}"))
    monkeypatch.setattr(socket, "gethostbyname", lambda h: wire.trip(f"gethostbyname {h!r}"))
    return wire
