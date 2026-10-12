"""The two engine interfaces the pipeline composes.

A PageReader turns one rendered page image into text (Tesseract, Cloud Vision) or into text plus
field values (a vision LLM). A FieldExtractor turns page text into field values (an LLM, or the
deterministic rule extractor). Each declares its network ``destination`` (None means it runs on
this machine), which is what the privacy tier is checked against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

from ..privacy import Destination
from ..records import FieldOutput, PageReading
from ..schema import DocSchema


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


class PageReader(Protocol):
    id: str  # shown in results
    fingerprint: str  # every setting that changes the answer; part of cache keys
    extracts_fields: bool
    render_dpi: int
    destination: Destination | None

    def read(self, image: Image.Image, page: int, pages_total: int,
             schema: DocSchema) -> PageReading: ...


class FieldExtractor(Protocol):
    id: str
    fingerprint: str
    destination: Destination | None
    can_repair: bool

    def extract(self, pages: Sequence[PageText], schema: DocSchema) -> FieldOutput: ...

    def repair(self, pages: Sequence[PageText], schema: DocSchema,
               previous: Mapping[str, Any], problems: Sequence[str]) -> FieldOutput: ...


def legibility_from(score: float | None, good: float, fair: float,
                    unknown: str = "fair") -> str:
    if score is None:
        return unknown
    return "good" if score >= good else "fair" if score >= fair else "poor"
