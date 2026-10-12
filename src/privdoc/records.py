"""Small value types shared by the engines and the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    ocr_pages: int = 0
    cost_usd: float | None = 0.0  # None: some part of it has no configured price

    def __add__(self, other: Usage) -> Usage:
        cost = (None if self.cost_usd is None or other.cost_usd is None
                else self.cost_usd + other.cost_usd)
        return Usage(self.llm_calls + other.llm_calls, self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens, self.ocr_pages + other.ocr_pages,
                     cost)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.cost_usd is not None:
            out["cost_usd"] = round(self.cost_usd, 6)
        return out


@dataclass
class PageReading:
    """What one engine made of one page."""

    page: int
    reader: str
    text: str
    legibility: str = "good"  # good | fair | poor
    fields: dict[str, Any] | None = None  # raw per-field answers, for engines that extract
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    cached: bool = False

    def to_cache(self) -> dict[str, Any]:
        return {"page": self.page, "reader": self.reader, "text": self.text,
                "legibility": self.legibility, "fields": self.fields,
                "usage": asdict(self.usage), "seconds": self.seconds}

    @classmethod
    def from_cache(cls, data: dict[str, Any]) -> PageReading:
        return cls(page=data["page"], reader=data["reader"], text=data["text"],
                   legibility=data.get("legibility", "good"), fields=data.get("fields"),
                   usage=Usage(**data.get("usage", {})), seconds=data.get("seconds", 0.0),
                   cached=True)


@dataclass
class FieldOutput:
    """Raw per-field answers from a text extractor, with what producing them cost."""

    fields: dict[str, Any]
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    cached: bool = False
    truncated: bool = False
