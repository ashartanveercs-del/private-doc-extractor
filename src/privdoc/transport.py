"""JSON-over-HTTPS for API calls (LLM, cloud OCR).

Only the EgressGate calls a Transport, so every request that can carry document content is
checked against the privacy tier first. The real transport never follows redirects: a 3xx is
returned as-is, so content cannot be bounced to a host the gate did not approve.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from ._version import __version__
from .errors import EngineError

MAX_RESPONSE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise EngineError("the service returned a response that is not JSON") from None

    def header(self, name: str) -> str | None:
        name = name.lower()
        for key, value in self.headers.items():
            if key.lower() == name:
                return value
        return None


class Transport(Protocol):
    def post_json(self, url: str, *, headers: Mapping[str, str], body: Any,
                  timeout: float) -> HttpResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None  # urllib then raises HTTPError carrying the 3xx status


def _read_capped(stream, cap: int) -> bytes:
    data = stream.read(cap + 1)
    if len(data) > cap:
        raise EngineError("the service's response was larger than the size limit")
    return data


class UrllibTransport:
    def __init__(self, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
        self._cap = max_response_bytes
        self._opener = urllib.request.build_opener(_NoRedirect)

    def post_json(self, url: str, *, headers: Mapping[str, str], body: Any,
                  timeout: float) -> HttpResponse:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": f"privdoc/{__version__}", **headers})
        host = urlsplit(url).hostname or "the service"
        try:
            with self._opener.open(request, timeout=timeout) as resp:
                return HttpResponse(resp.status, dict(resp.headers.items()),
                                    _read_capped(resp, self._cap))
        except urllib.error.HTTPError as err:
            with err:
                return HttpResponse(err.code, dict(err.headers.items()),
                                    _read_capped(err, self._cap))
        except urllib.error.URLError as err:
            reason = type(err.reason).__name__ if isinstance(err.reason, Exception) else "error"
            raise EngineError(f"could not reach {host} ({reason})") from None
        except (TimeoutError, OSError) as err:
            raise EngineError(f"could not reach {host} ({type(err).__name__})") from None
