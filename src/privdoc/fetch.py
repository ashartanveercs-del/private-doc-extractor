"""Download a document from a link without becoming an SSRF proxy.

A link handed to an extraction service is attacker-controlled input. Fetching it naively lets
the caller read the cloud metadata service (169.254.169.254), admin panels on localhost or
anything else on the private network. The fetcher therefore:

* accepts only http(s) URLs without credentials, whitespace, control characters or backslashes;
* refuses numeric host shorthands (``2130706433``, ``0x7f.1``, ``127.1``) whose meaning depends
  on which parser reads them;
* resolves the name once, refuses it if *any* address is not globally routable (including IPv4
  hidden inside IPv6: mapped, compatible, 6to4, Teredo and NAT64 forms), and then connects to
  exactly the address it checked, so a DNS answer cannot change between check and use;
* follows redirects by hand and repeats every check on every hop;
* refuses compressed bodies, caps the size whether or not it was declared, stops a server that
  trickles bytes, and accepts the body only if its bytes really are a supported document whose
  declared content type agrees.

Error messages name the host, never the full URL: signed links carry tokens in the query string.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urljoin, urlsplit

from ._version import __version__
from .errors import FetchError, UnsafeURL
from .sniff import DOCUMENT_KINDS, MEDIA_TYPES, sniff

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

REDIRECTS = frozenset({301, 302, 303, 307, 308})
GENERIC_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream",
                           "application/download", "application/force-download"})
_UNSAFE_CHARS = re.compile(r"[\x00-\x20\x7f\\]")
_NUMERIC_LABEL = re.compile(r"0x[0-9a-f]*|[0-9]+", re.IGNORECASE)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPAT = ipaddress.ip_network("::/96")


@dataclass(frozen=True)
class FetchPolicy:
    max_bytes: int = 25 * 1024 * 1024
    timeout: float = 30.0  # for the whole download, redirects included
    max_redirects: int = 5
    allowed_ports: frozenset[int] | None = frozenset({80, 443})  # None: any port
    allowed_kinds: frozenset[str] = DOCUMENT_KINDS
    # explicit exceptions to the public-address rule, e.g. a trusted intranet document server
    allowed_networks: tuple[IPNetwork, ...] = ()


@dataclass(frozen=True)
class OpenRequest:
    scheme: str
    host: str
    port: int
    ip: str
    target: str
    headers: dict[str, str]
    timeout: float


class OpenResponse(Protocol):
    status: int
    headers: Mapping[str, str]  # lower-case names

    def read(self, n: int) -> bytes: ...

    def close(self) -> None: ...


Opener = Callable[[OpenRequest], OpenResponse]
Resolver = Callable[[str, int], list[str]]


@dataclass
class Fetched:
    url: str
    final_url: str
    data: bytes
    kind: str
    content_type: str
    hosts: list[str] = field(default_factory=list)

    @property
    def final_host(self) -> str:
        return self.hosts[-1] if self.hosts else ""


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    found = [a for a in (ip.ipv4_mapped, ip.sixtofour) if a is not None]
    if ip.teredo is not None:
        found.append(ip.teredo[1])
    if ip in _NAT64 or (ip in _V4_COMPAT and int(ip) > 1):
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def is_public_address(ip: IPAddress) -> bool:
    """True only for globally routable unicast addresses, looking through IPv6 wrappers."""
    if ip.is_multicast or ip.is_unspecified or ip.is_reserved or not ip.is_global:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        return all(is_public_address(inner) for inner in _embedded_ipv4(ip))
    return True


def system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


class _Deadline:
    """Socket mixin: every send and receive waits at most until one fixed deadline.

    A socket timeout bounds a single call, and http.client makes many: a server that sends one
    byte just before each timeout keeps a download alive for hours. Setting the timeout to the
    time that is left before every call bounds the whole exchange, status line, headers, chunk
    sizes and body alike, and once the deadline has passed the next call fails at once.
    """

    __slots__ = ()
    deadline: float | None = None

    def _time_left(self) -> None:
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("the overall deadline has passed")
            self.settimeout(left)

    def recv_into(self, *args, **kwargs):
        self._time_left()
        return super().recv_into(*args, **kwargs)

    def recv(self, *args, **kwargs):
        self._time_left()
        return super().recv(*args, **kwargs)

    def sendall(self, *args, **kwargs):
        self._time_left()
        return super().sendall(*args, **kwargs)


class _DeadlineSocket(_Deadline, socket.socket):
    pass


class _DeadlineTLSSocket(_Deadline, ssl.SSLSocket):
    pass


class _PinnedConnection(http.client.HTTPConnection):
    """Connects to a pre-checked IP while keeping the host name for Host, SNI and TLS checks."""

    def __init__(self, host: str, port: int, *, ip: str, deadline: float,
                 tls: ssl.SSLContext | None) -> None:
        super().__init__(host, port, timeout=max(0.0, deadline - time.monotonic()))
        self._pinned_ip = ip
        self._deadline = deadline
        self._tls = tls
        if tls is not None:
            self.default_port = 443  # keeps "Host: name" instead of "Host: name:443"
            tls.sslsocket_class = _DeadlineTLSSocket

    def connect(self) -> None:
        ip = ipaddress.ip_address(self._pinned_ip)
        sock = _DeadlineSocket(socket.AF_INET6 if ip.version == 6 else socket.AF_INET,
                               socket.SOCK_STREAM)
        sock.deadline = self._deadline
        try:
            sock._time_left()  # the connect and the TLS handshake get only the time left too
            sock.connect((self._pinned_ip, self.port))
            if self._tls is not None:
                sock._time_left()
                sock = self._tls.wrap_socket(sock, server_hostname=self.host)
                sock.deadline = self._deadline
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class _Response:
    def __init__(self, conn: http.client.HTTPConnection, resp: http.client.HTTPResponse) -> None:
        self._conn = conn
        self._resp = resp
        self.status = resp.status
        self.headers = {k.lower(): v for k, v in resp.getheaders()}

    def read(self, n: int) -> bytes:
        return self._resp.read(n)

    def close(self) -> None:
        self._resp.close()
        self._conn.close()


def pinned_open(request: OpenRequest) -> OpenResponse:
    tls = ssl.create_default_context() if request.scheme == "https" else None
    conn = _PinnedConnection(request.host, request.port, ip=request.ip,
                             deadline=time.monotonic() + request.timeout, tls=tls)
    try:
        conn.request("GET", request.target, headers=request.headers)
        return _Response(conn, conn.getresponse())
    except BaseException:
        conn.close()
        raise


@dataclass(frozen=True)
class _Target:
    scheme: str
    host: str
    port: int
    target: str
    ips: list[str]


class SafeFetcher:
    def __init__(self, policy: FetchPolicy | None = None, *, resolver: Resolver = system_resolver,
                 opener: Opener = pinned_open, clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy or FetchPolicy()
        self.resolver = resolver
        self.opener = opener
        self.clock = clock

    def fetch(self, url: str) -> Fetched:
        deadline = self.clock() + self.policy.timeout
        current, hosts = url, []
        for _ in range(self.policy.max_redirects + 1):
            target = self.check(current)
            hosts.append(target.host)
            response = self._open(target, deadline)
            try:
                if response.status in REDIRECTS:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(f"{target.host} sent a redirect without a location")
                    current = urljoin(current, location.strip())
                    continue
                declared, data = self._body(response, target.host, deadline)
            finally:
                response.close()
            kind = sniff(data)
            if kind not in self.policy.allowed_kinds:
                raise FetchError(f"{target.host} did not return a supported document")
            if declared not in GENERIC_TYPES and declared not in MEDIA_TYPES.get(kind, ()):
                raise FetchError(f"{target.host} declared {declared}, which does not match the "
                                 f"content ({kind})")
            return Fetched(url, current, data, kind, declared, hosts)
        raise FetchError(f"too many redirects (more than {self.policy.max_redirects})")

    # -- checks --------------------------------------------------------------------------------

    def check(self, url: str) -> _Target:
        if _UNSAFE_CHARS.search(url):
            raise UnsafeURL("the link contains spaces, control characters or backslashes")
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            raise UnsafeURL("only http and https links can be fetched")
        if "@" in parts.netloc:
            raise UnsafeURL("links with embedded credentials are refused")
        host = (parts.hostname or "").lower()
        if not host:
            raise UnsafeURL("the link has no host")
        try:
            port = parts.port or (443 if scheme == "https" else 80)
        except ValueError:
            raise UnsafeURL("the link has an invalid port") from None
        allowed = self.policy.allowed_ports
        if allowed is not None and port not in allowed:
            raise UnsafeURL(f"port {port} is not allowed (allowed: {sorted(allowed)})")
        literal = self._literal_ip(host)
        if literal is not None:
            addresses = [literal]
        else:
            addresses = self._resolve(host, port)
        for ip in addresses:
            if not self._address_ok(ip):
                raise UnsafeURL(f"{host} points to a private or reserved address; "
                                "refusing to fetch it")
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        return _Target(scheme, host, port, target, [str(a) for a in addresses])

    @staticmethod
    def _literal_ip(host: str) -> IPAddress | None:
        try:
            return ipaddress.ip_address(host)
        except ValueError:
            pass
        labels = host.rstrip(".").split(".")
        if all(_NUMERIC_LABEL.fullmatch(label) for label in labels):
            raise UnsafeURL(f"numeric host {host!r} is not a standard address; refusing it")
        return None

    def _resolve(self, host: str, port: int) -> list[IPAddress]:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise UnsafeURL("the link has an invalid host name") from None
        try:
            answers = self.resolver(ascii_host, port)
        except OSError:
            raise FetchError(f"could not look up {host}") from None
        addresses = []
        for answer in answers:
            try:
                addresses.append(ipaddress.ip_address(answer.split("%")[0]))
            except ValueError:
                continue
        if not addresses:
            raise FetchError(f"could not look up {host}")
        return addresses

    def _address_ok(self, ip: IPAddress) -> bool:
        if any(ip in net for net in self.policy.allowed_networks if net.version == ip.version):
            return True
        return is_public_address(ip)

    # -- I/O -----------------------------------------------------------------------------------

    def _open(self, target: _Target, deadline: float) -> OpenResponse:
        headers = {"User-Agent": f"privdoc/{__version__}", "Accept-Encoding": "identity",
                   "Accept": "application/pdf, image/*;q=0.9, application/octet-stream;q=0.5"}
        last: Exception | None = None
        for ip in target.ips[:3]:
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            try:
                return self.opener(OpenRequest(target.scheme, target.host, target.port, ip,
                                               target.target, headers, remaining))
            except (OSError, http.client.HTTPException) as err:
                last = err
        if last is None or isinstance(last, TimeoutError):
            raise FetchError(f"the download from {target.host} timed out")
        raise FetchError(f"could not download from {target.host} ({type(last).__name__})")

    def _body(self, response: OpenResponse, host: str, deadline: float) -> tuple[str, bytes]:
        if response.status != 200:
            raise FetchError(f"{host} answered HTTP {response.status}")
        encoding = (response.headers.get("content-encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise FetchError(f"{host} sent a compressed response ({encoding}); refusing to "
                             "inflate it")
        declared = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        accepted = set().union(*(MEDIA_TYPES.get(k, ()) for k in self.policy.allowed_kinds))
        if declared not in GENERIC_TYPES and declared not in accepted:
            raise FetchError(f"{host} returned content type {declared or 'unknown'}, "
                             "not a supported document")
        cap = self.policy.max_bytes
        limit = f"the file from {host} is larger than the {cap / 1048576:g} MB size limit"
        length = response.headers.get("content-length")
        if length and length.strip().isdigit() and int(length) > cap:
            raise FetchError(limit)
        chunks, total = [], 0
        try:
            while True:
                if self.clock() > deadline:
                    raise FetchError(f"the download from {host} timed out")
                chunk = response.read(min(65536, cap + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > cap:
                    raise FetchError(limit)
        except TimeoutError:
            raise FetchError(f"the download from {host} timed out") from None
        except (OSError, http.client.HTTPException) as err:
            raise FetchError(f"the download from {host} failed ({type(err).__name__})") from None
        return declared, b"".join(chunks)
