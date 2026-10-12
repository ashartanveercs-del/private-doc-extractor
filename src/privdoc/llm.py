"""A minimal client for OpenAI-compatible chat completions with strict JSON-schema output.

Works with any server that implements ``POST {base}/chat/completions`` with
``response_format: {"type": "json_schema", ...}``: hosted APIs, or vLLM / other servers you run
yourself. Requests go through the EgressGate; a blocked send is a privacy error and is never
retried. API keys are never put in error messages, even if the API echoes them back.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .coerce import parse_json_object
from .errors import EngineError
from .privacy import Destination, EgressGate, Trust

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 30.0


@dataclass
class ChatResult:
    data: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int


class ChatClient:
    def __init__(self, gate: EgressGate, base_url: str, api_key: str = "", *,
                 trust: Trust = Trust.THIRD_PARTY, timeout: float = 120.0, max_retries: int = 2,
                 temperature: float | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.gate = gate
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.destination = Destination("LLM API", self.base_url, trust)
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.sleep = sleep

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def complete_json(self, *, model: str, system: str, user: str | list[dict[str, Any]],
                      schema_name: str, schema: dict[str, Any]) -> ChatResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema_name, "strict": True, "schema": schema}},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        url = f"{self.base_url}/chat/completions"

        for attempt in range(self.max_retries + 1):
            last_try = attempt == self.max_retries
            try:
                response = self.gate.post_json(url, headers=headers, body=body,
                                               timeout=self.timeout)
            except EngineError as err:  # network trouble; EgressBlocked is not an EngineError
                if last_try:
                    raise EngineError(self._redact(str(err))) from None
                self.sleep(1.5 * 2 ** attempt)
                continue
            if response.status == 200:
                return self._parse(response.json(), model)
            message = self._error_message(response)
            if response.status in RETRY_STATUSES and not last_try:
                self.sleep(self._retry_delay(response.header("retry-after"), attempt))
                continue
            raise EngineError(f"{self.destination.host} returned HTTP {response.status}"
                              + (f": {message}" if message else ""))
        raise AssertionError("unreachable")

    def _retry_delay(self, retry_after: str | None, attempt: int) -> float:
        try:
            return min(MAX_RETRY_AFTER, max(0.0, float(retry_after or "")))
        except ValueError:
            return 1.5 * 2 ** attempt

    def _error_message(self, response) -> str:
        try:
            error = response.json().get("error")
        except (EngineError, AttributeError):
            return ""
        text = error.get("message", "") if isinstance(error, dict) else str(error or "")
        return self._redact(" ".join(text.split()))[:200]

    def _parse(self, payload: Any, model: str) -> ChatResult:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise EngineError(f"{model} returned no answer")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise EngineError(f"{model} refused to read this page")
        if choice.get("finish_reason") == "length":
            raise EngineError(f"{model}'s answer was cut off (output token limit)")
        try:
            data = parse_json_object(message.get("content"))
        except ValueError:
            raise EngineError(f"{model}'s answer was not valid JSON") from None
        usage = payload.get("usage") or {}
        return ChatResult(data=data, model=str(payload.get("model") or model),
                          input_tokens=int(usage.get("prompt_tokens") or 0),
                          output_tokens=int(usage.get("completion_tokens") or 0))
