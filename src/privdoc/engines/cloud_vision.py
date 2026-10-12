"""OCR with Google Cloud Vision (DOCUMENT_TEXT_DETECTION) in your own cloud project.

This is the "own-cloud" engine: pages go to a service under your own account and agreement,
not to a shared AI provider. Authenticate with an API key (sent in the ``x-goog-api-key``
header, never in the URL) or an OAuth access token. The endpoint is configurable, e.g. for a
regional endpoint. Only text comes back, so fields are extracted afterwards by a text extractor.
"""

from __future__ import annotations

import base64
import time
from typing import Any

from PIL import Image

from ..document import downscale, encode_png
from ..errors import EngineError
from ..pricing import Prices
from ..privacy import Destination, EgressGate, Trust
from ..records import PageReading, Usage
from ..schema import DocSchema
from .base import legibility_from


def _page_confidence(annotation: dict[str, Any]) -> float | None:
    pages = annotation.get("pages") or []
    scores = [p["confidence"] for p in pages if isinstance(p.get("confidence"), int | float)]
    if not scores:
        scores = [b["confidence"] for p in pages for b in p.get("blocks") or []
                  if isinstance(b.get("confidence"), int | float)]
    return sum(scores) / len(scores) if scores else None


class CloudVisionReader:
    extracts_fields = False
    render_dpi = 300

    def __init__(self, gate: EgressGate, *, endpoint: str = "https://vision.googleapis.com",
                 api_key: str = "", access_token: str = "", quota_project: str = "",
                 lang: str = "en", prices: Prices | None = None, timeout: float = 60.0,
                 max_px: int = 4000) -> None:
        self.gate = gate
        self.endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._token = access_token
        self.quota_project = quota_project
        self.lang = lang
        self.prices = prices or Prices()
        self.timeout = timeout
        self.max_px = max_px
        self.destination = Destination("Cloud Vision OCR", self.endpoint, Trust.OWNER)
        self.id = f"cloud-ocr:{self.destination.host}"
        self.fingerprint = f"{self.id}:{lang}:{max_px}"

    def _redact(self, text: str) -> str:
        for secret in (self._api_key, self._token):
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    def _headers(self) -> dict[str, str]:
        if self._token:
            headers = {"Authorization": f"Bearer {self._token}"}
            if self.quota_project:
                headers["x-goog-user-project"] = self.quota_project
            return headers
        return {"x-goog-api-key": self._api_key}

    def read(self, image: Image.Image, page: int, pages_total: int,
             schema: DocSchema) -> PageReading:
        started = time.perf_counter()
        content = base64.b64encode(encode_png(downscale(image, self.max_px))).decode("ascii")
        body = {"requests": [{"image": {"content": content},
                              "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                              "imageContext": {"languageHints": [self.lang]}}]}
        response = self.gate.post_json(f"{self.endpoint}/v1/images:annotate",
                                       headers=self._headers(), body=body, timeout=self.timeout)
        if response.status != 200:
            raise EngineError(f"Cloud Vision returned HTTP {response.status} for page {page}")
        payload = response.json()
        item = ((payload.get("responses") if isinstance(payload, dict) else None) or [{}])[0]
        error = item.get("error")
        if error:
            message = error.get("message", "unknown error") if isinstance(error, dict) else error
            raise EngineError(f"Cloud Vision could not read page {page}: "
                              f"{self._redact(str(message))[:200]}")
        annotation = item.get("fullTextAnnotation") or {}
        text = str(annotation.get("text") or "").strip()
        confidence = _page_confidence(annotation)
        legibility = legibility_from(confidence, good=0.9, fair=0.75) if text else "poor"
        return PageReading(page=page, reader=self.id, text=text, legibility=legibility,
                           usage=Usage(ocr_pages=1, cost_usd=self.prices.ocr_cost(1)),
                           seconds=round(time.perf_counter() - started, 3))
