import pytest

from privdoc.errors import EgressBlocked, PrivacyViolation, UsageError
from privdoc.privacy import Destination, EgressGate, Tier, Trust, parse_tier
from tests.fakes import FakeTransport

LLM = Destination("LLM API", "https://api.llm.example.com/v1", Trust.THIRD_PARTY)
OCR = Destination("Cloud OCR", "https://ocr.example.com", Trust.OWNER)


@pytest.mark.parametrize("spelling, tier", [
    ("cloud", Tier.CLOUD), ("CLOUD", Tier.CLOUD), ("own-cloud", Tier.OWN_CLOUD),
    ("own_cloud", Tier.OWN_CLOUD), ("Own Cloud", Tier.OWN_CLOUD), ("owncloud", Tier.OWN_CLOUD),
    ("local", Tier.LOCAL), (" Local ", Tier.LOCAL),
])
def test_tier_spelling_is_forgiven(spelling, tier):
    assert parse_tier(spelling) is tier


def test_unknown_tier_is_refused_rather_than_guessed():
    with pytest.raises(UsageError, match="cloud, own-cloud or local"):
        parse_tier("private")


def test_each_tier_trusts_a_shrinking_set_of_destinations():
    assert Tier.CLOUD.allowed_trust == {Trust.THIRD_PARTY, Trust.OWNER}
    assert Tier.OWN_CLOUD.allowed_trust == {Trust.OWNER}
    assert Tier.LOCAL.allowed_trust == frozenset()


def test_own_cloud_refuses_a_third_party_destination_before_anything_is_sent():
    transport = FakeTransport()
    gate = EgressGate(Tier.OWN_CLOUD, transport)
    with pytest.raises(PrivacyViolation, match="own-cloud"):
        gate.allow(LLM)
    gate.allow(OCR)
    assert gate.allowed_hosts == ["ocr.example.com"]
    assert transport.requests == []


def test_local_tier_refuses_every_destination():
    gate = EgressGate(Tier.LOCAL, FakeTransport())
    for dest in (LLM, OCR):
        with pytest.raises(PrivacyViolation):
            gate.allow(dest)


def test_documents_never_go_to_a_third_party_over_plain_http():
    gate = EgressGate(Tier.CLOUD, FakeTransport())
    with pytest.raises(PrivacyViolation, match="https"):
        gate.allow(Destination("LLM API", "http://api.llm.example.com/v1", Trust.THIRD_PARTY))
    # a server in your own network may be plain http
    gate.allow(Destination("Own LLM", "http://10.0.0.5:8000/v1", Trust.OWNER))


def test_allowed_send_goes_through_and_is_recorded():
    transport = FakeTransport(lambda url, headers, body: (200, {"ok": True}))
    gate = EgressGate(Tier.CLOUD, transport)
    gate.allow(LLM)
    response = gate.post_json("https://API.llm.example.com/v1/chat/completions",
                              headers={"Authorization": "Bearer k"}, body={"x": 1}, timeout=5)
    assert response.status == 200 and response.json() == {"ok": True}
    assert [r.url for r in transport.requests] == [
        "https://API.llm.example.com/v1/chat/completions"]
    assert gate.data_sent_to == ["LLM API (api.llm.example.com, third-party)"]


@pytest.mark.parametrize("url", [
    "https://evil.example.net/v1/chat/completions",   # another host
    "https://api.llm.example.com:8443/v1/chat",       # same host, another port
    "http://api.llm.example.com/v1/chat",             # same host, downgraded scheme
])
def test_sends_anywhere_else_are_blocked_and_never_reach_the_transport(url):
    transport = FakeTransport()
    gate = EgressGate(Tier.CLOUD, transport)
    gate.allow(LLM)
    with pytest.raises(EgressBlocked):
        gate.post_json(url, headers={}, body={}, timeout=5)
    assert transport.requests == []
    assert gate.data_sent_to == []


def test_local_gate_blocks_sends_even_with_a_transport_attached():
    transport = FakeTransport()
    gate = EgressGate(Tier.LOCAL, transport)
    with pytest.raises(EgressBlocked, match="local"):
        gate.post_json("https://api.llm.example.com/v1/chat", headers={}, body={}, timeout=5)
    assert transport.requests == []


def test_a_send_is_recorded_before_the_request_because_a_failure_may_come_after_upload():
    gate = EgressGate(Tier.CLOUD, FakeTransport(lambda u, h, b: ConnectionResetError("reset")))
    gate.allow(LLM)
    with pytest.raises(ConnectionResetError):
        gate.post_json("https://api.llm.example.com/v1/chat", headers={}, body={}, timeout=5)
    assert gate.data_sent_to == ["LLM API (api.llm.example.com, third-party)"]
