import json

import pytest

from privdoc.errors import EngineError
from privdoc.transport import HttpResponse, UrllibTransport
from tests.localserver import LocalServer, reply


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.stop()


def test_posts_json_and_returns_status_headers_and_body(server):
    server.routes["/v1/echo"] = reply(200, b'{"ok": true}', {"Content-Type": "application/json"})
    response = UrllibTransport().post_json(server.url("/v1/echo"), headers={"X-Api-Key": "k1"},
                                           body={"page": 1}, timeout=5)
    assert response.status == 200
    assert response.json() == {"ok": True}
    seen = server.seen[0]
    assert seen.method == "POST"
    assert json.loads(seen.body) == {"page": 1}
    assert seen.headers["X-Api-Key"] == "k1"
    assert seen.headers["Content-Type"] == "application/json"


def test_error_statuses_are_returned_not_raised(server):
    server.routes["/v1/busy"] = reply(429, b'{"error": "slow down"}', {"Retry-After": "2"})
    response = UrllibTransport().post_json(server.url("/v1/busy"), headers={}, body={}, timeout=5)
    assert response.status == 429
    assert response.header("retry-after") == "2"


def test_redirects_are_never_followed_so_content_cannot_be_bounced_elsewhere(server):
    server.routes["/v1/moved"] = reply(307, b"", {"Location": server.url("/v1/elsewhere")})
    server.routes["/v1/elsewhere"] = reply(200, b"{}")
    response = UrllibTransport().post_json(server.url("/v1/moved"), headers={}, body={"secret": 1},
                                           timeout=5)
    assert response.status == 307
    assert [s.path for s in server.seen] == ["/v1/moved"]


def test_oversized_responses_are_cut_off(server):
    server.routes["/v1/huge"] = reply(200, b"x" * 5000)
    with pytest.raises(EngineError, match="size limit"):
        UrllibTransport(max_response_bytes=1000).post_json(server.url("/v1/huge"), headers={},
                                                           body={}, timeout=5)


def test_connection_failures_become_a_short_message_naming_only_the_host():
    gone = LocalServer()
    url = gone.url("/v1/x?token=supersecret")
    gone.stop()  # nothing listens on that port any more
    with pytest.raises(EngineError) as info:
        UrllibTransport().post_json(url, headers={}, body={}, timeout=2)
    assert "127.0.0.1" in str(info.value)
    assert "supersecret" not in str(info.value)


def test_non_json_body_is_a_clean_engine_error():
    with pytest.raises(EngineError, match="not JSON"):
        HttpResponse(200, {}, b"<html>").json()
