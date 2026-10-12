import ipaddress
from dataclasses import dataclass, field

import pytest

from privdoc.errors import FetchError, UnsafeURL
from privdoc.fetch import FetchPolicy, OpenRequest, SafeFetcher, is_public_address
from tests.localserver import LocalServer, chunked, reply

PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
PUBLIC = "93.184.215.14"  # example.com-style global addresses; nothing ever connects to them
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"


class FakeResolver:
    def __init__(self, table: dict[str, list[str]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append(host)
        if host not in self.table:
            raise OSError("no such host")
        return self.table[host]


@dataclass
class FakeResponse:
    status: int = 200
    headers: dict[str, str] = field(default_factory=lambda: {"content-type": "application/pdf"})
    body: bytes = PDF
    read_bytes: int = 0
    closed: bool = False

    def read(self, n: int) -> bytes:
        chunk = self.body[self.read_bytes:self.read_bytes + n]
        self.read_bytes += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses  # keyed by "host/target"
        self.requests: list[OpenRequest] = []

    def __call__(self, request: OpenRequest) -> FakeResponse:
        self.requests.append(request)
        return self.responses[f"{request.host}{request.target}"]


def make(responses=None, table=None, **policy):
    resolver = FakeResolver(table or {"docs.example.com": [PUBLIC]})
    opener = FakeOpener(responses or {"docs.example.com/invoice.pdf": FakeResponse()})
    return SafeFetcher(FetchPolicy(**policy), resolver=resolver, opener=opener), resolver, opener


def test_fetches_a_public_document_and_pins_the_connection_to_the_checked_address():
    fetcher, resolver, opener = make()
    got = fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert got.data == PDF and got.kind == "pdf"
    assert got.final_host == "docs.example.com"
    assert resolver.calls == ["docs.example.com"]  # resolved once, then pinned
    request = opener.requests[0]
    assert (request.host, request.ip, request.port, request.scheme) == (
        "docs.example.com", PUBLIC, 443, "https")
    assert request.headers["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://docs.example.com/invoice.pdf", "gopher://docs.example.com/",
    "data:application/pdf;base64,JVBERi0xLjQ=", "javascript:alert(1)", "//docs.example.com/x",
])
def test_only_http_and_https_links_are_fetched(url):
    fetcher, resolver, opener = make()
    with pytest.raises(UnsafeURL):
        fetcher.fetch(url)
    assert resolver.calls == [] and opener.requests == []


@pytest.mark.parametrize("url", [
    "https://user:pass@docs.example.com/invoice.pdf",
    "https://docs.example.com@10.0.0.1/invoice.pdf",
    "https://docs.example.com/a b.pdf",
    "https://docs.example.com/inv\x00oice.pdf",
    "https://docs.example.com\\@10.0.0.1/invoice.pdf",
    "https://docs.example.com/invoice.pdf\r\nHost: internal",
])
def test_credentials_whitespace_control_characters_and_backslashes_are_refused(url):
    fetcher, resolver, opener = make()
    with pytest.raises(UnsafeURL):
        fetcher.fetch(url)
    assert opener.requests == []


@pytest.mark.parametrize("host", ["2130706433", "0x7f000001", "0x7f.1", "127.1", "0177.0.0.1",
                                  "017700000001"])
def test_numeric_host_shorthands_are_refused_without_a_lookup(host):
    fetcher, resolver, opener = make()
    with pytest.raises(UnsafeURL, match="numeric"):
        fetcher.fetch(f"http://{host}/invoice.pdf")
    assert resolver.calls == [] and opener.requests == []


PRIVATE_LITERALS = [
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.0.10", "169.254.169.254", "100.64.0.1",
    "0.0.0.0", "224.0.0.1", "255.255.255.255", "[::1]", "[::]", "[fe80::1]", "[fc00::1]",
    "[ff02::1]", "[::ffff:127.0.0.1]", "[::ffff:169.254.169.254]", "[64:ff9b::7f00:1]",
    "[2002:7f00:1::1]", "[::7f00:1]", "[2001:0:4136:e378:8000:63bf:80ff:fffe]",
]


@pytest.mark.parametrize("host", PRIVATE_LITERALS)
def test_private_loopback_link_local_and_disguised_addresses_are_refused(host):
    fetcher, resolver, opener = make()
    with pytest.raises(UnsafeURL, match="private or reserved"):
        fetcher.fetch(f"http://{host}/latest/meta-data/")
    assert opener.requests == []


@pytest.mark.parametrize("addr", [PUBLIC, PUBLIC_V6, "93.184.216.34"])
def test_global_addresses_are_public(addr):
    assert is_public_address(ipaddress.ip_address(addr))


def test_a_name_that_resolves_to_a_private_address_is_refused():
    fetcher, _, opener = make(table={"intranet.example.com": ["10.0.0.7"]})
    with pytest.raises(UnsafeURL, match="intranet.example.com"):
        fetcher.fetch("https://intranet.example.com/invoice.pdf")
    assert opener.requests == []


def test_one_private_answer_among_public_ones_is_enough_to_refuse():
    fetcher, _, opener = make(table={"docs.example.com": [PUBLIC, "127.0.0.1"]})
    with pytest.raises(UnsafeURL):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert opener.requests == []


def test_ports_outside_the_allow_list_are_refused():
    fetcher, _, opener = make()
    with pytest.raises(UnsafeURL, match="port 8443"):
        fetcher.fetch("https://docs.example.com:8443/invoice.pdf")
    assert opener.requests == []


def test_every_redirect_hop_is_checked_again():
    responses = {
        "docs.example.com/invoice.pdf": FakeResponse(
            302, {"location": "http://169.254.169.254/latest/meta-data/"}, b""),
    }
    fetcher, _, opener = make(responses)
    with pytest.raises(UnsafeURL):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert len(opener.requests) == 1


def test_a_redirect_to_a_name_that_resolves_privately_is_refused():
    responses = {"docs.example.com/invoice.pdf": FakeResponse(
        301, {"location": "https://cache.example.com/x.pdf"}, b"")}
    fetcher, resolver, opener = make(
        responses, table={"docs.example.com": [PUBLIC], "cache.example.com": ["192.168.1.5"]})
    with pytest.raises(UnsafeURL):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert resolver.calls == ["docs.example.com", "cache.example.com"]
    assert len(opener.requests) == 1


def test_relative_redirects_resolve_against_the_current_url():
    responses = {
        "docs.example.com/invoice.pdf": FakeResponse(302, {"location": "/files/v2.pdf"}, b""),
        "docs.example.com/files/v2.pdf": FakeResponse(),
    }
    fetcher, _, opener = make(responses)
    got = fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert got.final_url == "https://docs.example.com/files/v2.pdf"
    assert [r.target for r in opener.requests] == ["/invoice.pdf", "/files/v2.pdf"]


def test_redirect_loops_are_capped():
    responses = {"docs.example.com/invoice.pdf": FakeResponse(
        302, {"location": "/invoice.pdf"}, b"")}
    fetcher, _, opener = make(responses, max_redirects=3)
    with pytest.raises(FetchError, match="redirect"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert len(opener.requests) == 4


def test_declared_size_over_the_cap_is_refused_before_reading():
    big = FakeResponse(headers={"content-type": "application/pdf", "content-length": "999999"})
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": big}, max_bytes=1000)
    with pytest.raises(FetchError, match="size limit"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert big.read_bytes == 0 and big.closed


def test_undeclared_size_is_cut_off_while_streaming():
    big = FakeResponse(body=PDF + b"0" * 50_000)
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": big}, max_bytes=4096)
    with pytest.raises(FetchError, match="size limit"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert big.read_bytes <= 4096 + 1


def test_an_html_page_is_refused_by_its_content_type_without_reading_it():
    page = FakeResponse(headers={"content-type": "text/html; charset=utf-8"}, body=b"<html>")
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": page})
    with pytest.raises(FetchError, match="content type"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")
    assert page.read_bytes == 0


def test_a_generic_content_type_is_fine_when_the_bytes_are_a_document():
    blob = FakeResponse(headers={"content-type": "application/octet-stream"})
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": blob})
    assert fetcher.fetch("https://docs.example.com/invoice.pdf").kind == "pdf"


def test_a_declared_type_that_does_not_match_the_bytes_is_refused():
    liar = FakeResponse(headers={"content-type": "image/png"})
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": liar})
    with pytest.raises(FetchError, match="does not match"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")


def test_bytes_that_are_not_a_supported_document_are_refused():
    fake_pdf = FakeResponse(body=b"<html><body>please log in</body></html>")
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": fake_pdf})
    with pytest.raises(FetchError, match="supported"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")


def test_compressed_responses_are_refused_rather_than_inflated():
    gz = FakeResponse(headers={"content-type": "application/pdf", "content-encoding": "gzip"})
    fetcher, _, _ = make({"docs.example.com/invoice.pdf": gz})
    with pytest.raises(FetchError, match="compressed"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")


def test_errors_name_the_host_but_never_the_signed_query_string():
    responses = {"docs.example.com/invoice.pdf?sig=topsecret": FakeResponse(404, {}, b"")}
    fetcher, _, _ = make(responses)
    with pytest.raises(FetchError) as info:
        fetcher.fetch("https://docs.example.com/invoice.pdf?sig=topsecret")
    assert "404" in str(info.value) and "docs.example.com" in str(info.value)
    assert "topsecret" not in str(info.value)


def test_the_overall_deadline_stops_a_trickling_server():
    class Slow(FakeResponse):
        def read(self, n):
            clock.now += 10
            return super().read(1)

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    resolver = FakeResolver({"docs.example.com": [PUBLIC]})
    opener = FakeOpener({"docs.example.com/invoice.pdf": Slow()})
    fetcher = SafeFetcher(FetchPolicy(timeout=30), resolver=resolver, opener=opener, clock=clock)
    with pytest.raises(FetchError, match="timed out"):
        fetcher.fetch("https://docs.example.com/invoice.pdf")


# --- real sockets on 127.0.0.1 ----------------------------------------------------------------

LOOPBACK_OK = FetchPolicy(allowed_ports=None,
                          allowed_networks=(ipaddress.ip_network("127.0.0.1/32"),))


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.stop()


def test_real_fetch_connects_to_the_pinned_address_and_keeps_the_host_name(server):
    server.routes["/invoice.pdf"] = reply(200, PDF, {"Content-Type": "application/pdf"})
    fetcher = SafeFetcher(LOOPBACK_OK, resolver=FakeResolver({"docs.test": ["127.0.0.1"]}))
    got = fetcher.fetch(f"http://docs.test:{server.port}/invoice.pdf")
    assert got.data == PDF
    assert server.seen[0].headers["Host"] == f"docs.test:{server.port}"


def test_real_fetch_refuses_loopback_under_the_default_policy(server):
    server.routes["/invoice.pdf"] = reply(200, PDF, {"Content-Type": "application/pdf"})
    fetcher = SafeFetcher(FetchPolicy(allowed_ports=None))
    with pytest.raises(UnsafeURL):
        fetcher.fetch(server.url("/invoice.pdf"))
    assert server.seen == []


def test_real_redirect_into_link_local_space_is_refused(server):
    server.routes["/go"] = reply(302, b"", {"Location": "http://169.254.169.254/latest/meta-data/"})
    fetcher = SafeFetcher(LOOPBACK_OK)
    with pytest.raises(UnsafeURL):
        fetcher.fetch(server.url("/go"))
    assert [s.path for s in server.seen] == ["/go"]


def test_real_chunked_download_is_cut_at_the_cap(server):
    server.routes["/big.pdf"] = chunked([PDF] + [b"0" * 4096] * 50, "application/pdf")
    policy = FetchPolicy(allowed_ports=None, max_bytes=10_000,
                         allowed_networks=(ipaddress.ip_network("127.0.0.1/32"),))
    with pytest.raises(FetchError, match="size limit"):
        SafeFetcher(policy).fetch(server.url("/big.pdf"))


def test_proxy_settings_are_ignored_so_the_pinned_address_is_what_gets_connected(
        server, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # nothing listens there
    server.routes["/invoice.pdf"] = reply(200, PDF, {"Content-Type": "application/pdf"})
    fetcher = SafeFetcher(LOOPBACK_OK, resolver=FakeResolver({"docs.test": ["127.0.0.1"]}))
    assert fetcher.fetch(f"http://docs.test:{server.port}/invoice.pdf").data == PDF
