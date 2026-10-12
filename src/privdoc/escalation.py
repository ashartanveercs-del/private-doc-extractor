"""Which pages deserve a second, more expensive reading?

Escalation is per page, never per document: re-reading a 20-page contract because one date was
smudged would multiply the cost for nothing. A page is re-read with the next rung of its ladder
when, in order of priority:

  0  its first reading failed (an API error, a timeout, a crashed OCR process)
  1  the engine reported it as poorly legible
  2  a field taken from it is unreadable, invalid, low-confidence or not in the page text
  3  a required field is missing and the page was OCR'd with less than good legibility

Only pages that have a next rung qualify, and at most ``budget`` pages are re-read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .fields import FieldResult
from .routing import Route
from .schema import DocSchema


@dataclass(frozen=True)
class PageView:
    number: int
    route: Route | None
    status: str
    legibility: str | None
    has_next_rung: bool


def _field_problem(result: FieldResult) -> str | None:
    if result.status == "unreadable":
        return "unreadable"
    if result.status == "invalid":
        return "failed validation"
    if result.status == "found" and result.grounded is False:
        return "not found in the page text"
    if result.status == "found" and result.confidence == "low":
        return "low confidence"
    return None


def choose_pages(pages: Sequence[PageView], results: Mapping[str, FieldResult],
                 schema: DocSchema, budget: int) -> list[tuple[int, str]]:
    by_number = {p.number: p for p in pages}
    wanted: dict[int, tuple[int, str]] = {}

    def want(number: int | None, priority: int, reason: str) -> None:
        page = by_number.get(number) if number is not None else None
        if page is None or not page.has_next_rung:
            return
        if number not in wanted or priority < wanted[number][0]:
            wanted[number] = (priority, reason)

    for page in pages:
        if page.status == "failed":
            want(page.number, 0, "the first reading failed")
        elif page.status == "read" and page.legibility == "poor":
            want(page.number, 1, "poor legibility")
    for spec in schema.fields:
        result = results[spec.name]
        problem = _field_problem(result)
        if problem:
            want(result.page, 2, f"{spec.name}: {problem}")
    missing = [s.name for s in schema.fields
               if s.required and results[s.name].status in ("not_found", "not_checked")]
    if missing:
        for page in pages:
            if page.route is Route.OCR and page.status == "read" and page.legibility != "good":
                want(page.number, 3, f"required field {missing[0]} is missing")
    ranked = sorted(wanted.items(), key=lambda item: (item[1][0], item[0]))
    return [(number, reason) for number, (_, reason) in ranked[:max(0, budget)]]
