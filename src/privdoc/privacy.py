"""Privacy tiers, and the gate that enforces them.

A tier is a statement about where document content may go:

  cloud      third-party AI services and services in your own cloud account
  own-cloud  only services that run under your own cloud account or on your own servers
  local      nowhere: no network call is made at all

The tier is enforced twice. When the engine plan is built, every engine's destination is
checked and a disallowed one is refused before the document is even opened. At send time,
every request that can carry document content goes through `EgressGate.post_json`, which only
lets through requests to the exact origins (scheme, host, port) approved for the plan. A bug
that routes content anywhere else fails closed instead of leaking.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from .errors import EgressBlocked, PrivacyViolation, UsageError
from .transport import HttpResponse, Transport


class Trust(StrEnum):
    THIRD_PARTY = "third-party"  # a shared AI provider's API
    OWNER = "owner"  # an account, deployment or server you control


class Tier(StrEnum):
    CLOUD = "cloud"
    OWN_CLOUD = "own-cloud"
    LOCAL = "local"

    @property
    def allowed_trust(self) -> frozenset[Trust]:
        return _ALLOWED[self]

    @property
    def summary(self) -> str:
        return _SUMMARY[self]


_ALLOWED = {
    Tier.CLOUD: frozenset({Trust.THIRD_PARTY, Trust.OWNER}),
    Tier.OWN_CLOUD: frozenset({Trust.OWNER}),
    Tier.LOCAL: frozenset(),
}
_SUMMARY = {
    Tier.CLOUD: "may send page images and text to third-party AI APIs",
    Tier.OWN_CLOUD: "only sends pages to services in your own cloud account",
    Tier.LOCAL: "nothing leaves this machine",
}
_ALIASES = {"cloud": Tier.CLOUD, "own-cloud": Tier.OWN_CLOUD, "owncloud": Tier.OWN_CLOUD,
            "local": Tier.LOCAL}


def parse_tier(value: str) -> Tier:
    key = "-".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
    try:
        return _ALIASES[key]
    except KeyError:
        raise UsageError(f"unknown privacy tier {value!r}; use cloud, own-cloud or local") from None


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port or (443 if scheme == "https" else 80)
    return scheme, (parts.hostname or "").lower(), port


@dataclass(frozen=True)
class Destination:
    """A service that may receive document content."""

    name: str
    base_url: str
    trust: Trust

    @property
    def host(self) -> str:
        return (urlsplit(self.base_url).hostname or "").lower()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.host}, {self.trust})"


def permits(tier: Tier, destination: Destination) -> bool:
    return destination.trust in tier.allowed_trust


class EgressGate:
    """The only path by which document content may leave the process."""

    def __init__(self, tier: Tier, transport: Transport | None = None) -> None:
        self.tier = tier
        self._transport = transport
        self._allowed: dict[tuple[str, str, int], Destination] = {}
        self._sent: list[str] = []
        self._lock = threading.Lock()

    def allow(self, destination: Destination) -> None:
        if not permits(self.tier, destination):
            raise PrivacyViolation(
                f"{destination.name} at {destination.host} is a {destination.trust} service; "
                f"the {self.tier} tier {self.tier.summary}")
        scheme = urlsplit(destination.base_url).scheme.lower()
        if scheme not in ("https", "http") or (
                scheme == "http" and destination.trust is Trust.THIRD_PARTY):
            raise PrivacyViolation(
                f"{destination.name} must use https: documents are never sent to a third party "
                "over plain http")
        self._allowed[_origin(destination.base_url)] = destination

    @property
    def allowed_hosts(self) -> list[str]:
        return sorted({d.host for d in self._allowed.values()})

    @property
    def destinations(self) -> list[str]:
        return list(dict.fromkeys(d.label for d in self._allowed.values()))

    @property
    def data_sent_to(self) -> list[str]:
        with self._lock:
            return list(self._sent)

    def post_json(self, url: str, *, headers: Mapping[str, str], body: Any,
                  timeout: float) -> HttpResponse:
        if self.tier is Tier.LOCAL:
            raise EgressBlocked("the local tier makes no network calls")
        destination = self._allowed.get(_origin(url))
        if destination is None:
            host = urlsplit(url).hostname or "?"
            raise EgressBlocked(f"{host} is not an approved destination for the {self.tier} tier")
        with self._lock:
            # recorded before the request: a failure can come after the upload
            if destination.label not in self._sent:
                self._sent.append(destination.label)
        transport = self._transport
        if transport is None:
            from .transport import UrllibTransport

            transport = self._transport = UrllibTransport()
        return transport.post_json(url, headers=headers, body=body, timeout=timeout)
