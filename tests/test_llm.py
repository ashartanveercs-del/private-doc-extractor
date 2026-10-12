import pytest

from privdoc.errors import EgressBlocked, EngineError
from privdoc.llm import ChatClient
from privdoc.privacy import EgressGate, Tier, Trust
from privdoc.transport import HttpResponse
from tests.fakes import FakeTransport, chat_reply

BASE = "https://llm.example.com/v1"
KEY = "sk-test-0123456789"
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
          "additionalProperties": False}


def make(handler, *, tier=Tier.CLOUD, api_key=KEY, **kw):
    transport = FakeTransport(handler)
    gate = EgressGate(tier, transport)
    sleeps: list[float] = []
    client = ChatClient(gate, BASE, api_key, sleep=sleeps.append, **kw)
    if tier is not Tier.LOCAL:
        gate.allow(client.destination)
    return client, transport, sleeps


def call(client):
    return client.complete_json(model="model-a", system="be strict", user="read this",
                                schema_name="document_fields", schema=SCHEMA)


def test_request_uses_strict_json_schema_output():
    client, transport, _ = make(lambda u, h, b: (200, chat_reply({"a": 1})))
    result = call(client)
    sent = transport.requests[0]
    assert sent.url == f"{BASE}/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {KEY}"
    assert sent.body["model"] == "model-a"
    assert sent.body["messages"] == [{"role": "system", "content": "be strict"},
                                     {"role": "user", "content": "read this"}]
    assert sent.body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "document_fields", "strict": True, "schema": SCHEMA}}
    assert "temperature" not in sent.body
    assert result.data == {"a": 1}
    assert (result.input_tokens, result.output_tokens) == (1000, 200)


def test_no_key_means_no_authorization_header_and_temperature_is_opt_in():
    client, transport, _ = make(lambda u, h, b: (200, chat_reply({"a": 1})), api_key="",
                                temperature=0.0)
    call(client)
    assert "Authorization" not in transport.requests[0].headers
    assert transport.requests[0].body["temperature"] == 0.0


def test_fenced_json_content_is_accepted():
    client, _, _ = make(lambda u, h, b: (200, chat_reply('```json\n{"a": 2}\n```')))
    assert call(client).data == {"a": 2}


@pytest.mark.parametrize("reply, message", [
    (chat_reply({}, refusal="I can't help with that"), "refused"),
    (chat_reply('{"a": 1', finish_reason="length"), "cut off"),
    (chat_reply("the answer is three"), "not valid JSON"),
    ({"choices": []}, "no answer"),
])
def test_unusable_answers_are_engine_errors(reply, message):
    client, _, _ = make(lambda u, h, b: (200, reply))
    with pytest.raises(EngineError, match=message):
        call(client)


def test_rate_limits_are_retried_after_the_advertised_delay():
    replies = [HttpResponse(429, {"Retry-After": "2"}, b'{"error": {"message": "slow down"}}'),
               (200, chat_reply({"a": 3}))]
    client, transport, sleeps = make(lambda u, h, b: replies.pop(0))
    assert call(client).data == {"a": 3}
    assert sleeps == [2.0] and len(transport.requests) == 2


def test_server_errors_are_retried_with_backoff_then_reported():
    client, transport, sleeps = make(lambda u, h, b: (503, {"error": {"message": "overloaded"}}),
                                     max_retries=2)
    with pytest.raises(EngineError, match="HTTP 503"):
        call(client)
    assert len(transport.requests) == 3 and sleeps == [1.5, 3.0]


def test_client_errors_are_not_retried_and_never_echo_the_key():
    client, transport, _ = make(lambda u, h, b: (401, {"error": {
        "message": f"Incorrect API key provided: {KEY}"}}))
    with pytest.raises(EngineError) as info:
        call(client)
    assert len(transport.requests) == 1
    assert "HTTP 401" in str(info.value) and KEY not in str(info.value)


def test_network_errors_are_retried():
    replies = [EngineError("could not reach llm.example.com (ConnectionResetError)"),
               (200, chat_reply({"a": 4}))]
    client, _, sleeps = make(lambda u, h, b: replies.pop(0))
    assert call(client).data == {"a": 4} and sleeps == [1.5]


def test_a_blocked_egress_is_never_retried_or_softened():
    client, transport, _ = make(lambda u, h, b: (200, chat_reply({"a": 1})), tier=Tier.LOCAL)
    with pytest.raises(EgressBlocked):
        call(client)
    assert transport.requests == []


def test_destination_trust_is_configurable():
    client, _, _ = make(lambda u, h, b: (200, {}), trust=Trust.OWNER)
    assert client.destination.trust is Trust.OWNER and client.destination.host == "llm.example.com"
