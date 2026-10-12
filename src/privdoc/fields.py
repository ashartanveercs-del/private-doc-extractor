"""What one field's value is, how sure we are, and what we may honestly say about it.

Field statuses in the output:

  found        a value that passed coercion, validation and (ideally) grounding
  unreadable   the field is on the page but could not be read reliably; ``partial`` holds
               whatever was made out. Ask a human; never fill it in.
  invalid      a value was read but fails the schema even after repair; ``raw`` holds it
  not_found    every page of the document was read and the field is not on any of them
  not_checked  not on the pages that were read, but some pages were not read (failed,
               skipped, capped or no OCR engine). Never report this as "not on the document".
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .coerce import coerce
from .grounding import is_grounded, normalize_text
from .schema import CONFIDENCES, DocSchema, FieldSpec

CONFIDENCE_RANK = {name: rank for rank, name in enumerate(reversed(CONFIDENCES))}
_ILLEGIBLE_ONLY = re.compile(r"^\W*(?:\[?illegible\]?|unreadable|unclear|\?+)\W*$", re.IGNORECASE)
_MAX_TEXT = 300


@dataclass
class Candidate:
    """One reading of one field, from one page, by one engine."""

    field: str
    page: int | None
    source: str
    value: Any = None
    raw: Any = None
    confidence: str = "none"
    evidence: str | None = None
    unreadable: bool = False
    partial: str | None = None
    grounded: bool | None = None
    issues: tuple[str, ...] = ()
    escalated: bool = False
    alternatives: tuple[Candidate, ...] = ()  # other readings of this field on the same page

    @property
    def usable(self) -> bool:
        return self.value is not None and not self.issues

    @property
    def seen(self) -> bool:
        """Something was on the page for this field, even if it could not be used."""
        return self.value is not None or self.unreadable

    @property
    def trusted(self) -> bool:
        return (self.usable and self.grounded is not False
                and CONFIDENCE_RANK[self.confidence] >= CONFIDENCE_RANK["medium"])


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def normalize_candidate(raw: Mapping[str, Any] | None, spec: FieldSpec, schema: DocSchema, *,
                        page: int | None, page_text: str, source: str,
                        date_order: str = "MDY") -> Candidate:
    """Enforce the output's promises on one reported field."""
    raw = raw if isinstance(raw, Mapping) else {}
    value = raw.get("value")
    unreadable = raw.get("unreadable") is True
    confidence = raw.get("confidence")
    confidence = confidence if confidence in CONFIDENCE_RANK else "low"
    evidence = raw.get("evidence")
    # grounded against the whole quote; trimmed only for the output (a cut can split a word)
    evidence = (" ".join(evidence.split())
                if isinstance(evidence, str) and evidence.strip() else None)
    partial = None
    if isinstance(value, str):
        text = value.strip()
        if _ILLEGIBLE_ONLY.match(text):
            unreadable, value = True, None
        elif "[illegible]" in text.lower():
            unreadable, partial, value = True, text[:_MAX_TEXT], None
    if unreadable:
        if value is not None and partial is None:
            partial = _as_text(value)[:_MAX_TEXT]
        value = None

    original = value
    issues: tuple[str, ...] = ()
    grounded = None
    if value is not None:
        value = coerce(spec.schema, value, date_order)
    if value is not None:
        issues = tuple(str(i) for i in schema.validate_field(spec.name, value))
        grounded = is_grounded(value, evidence, page_text, spec.schema, date_order)
        if not grounded or confidence == "none":
            confidence = "low"
    else:
        confidence = "none"
    return Candidate(field=spec.name, page=page, source=source, value=value, raw=original,
                     confidence=confidence, evidence=evidence[:_MAX_TEXT] if evidence else None,
                     unreadable=unreadable, partial=partial, grounded=grounded, issues=issues)


def _partial_of(c: Candidate) -> str | None:
    if c.partial:
        return c.partial
    if c.value is not None:
        return _as_text(c.value)[:_MAX_TEXT]
    return None


def _trust(c: Candidate) -> tuple[bool, int]:
    """How far a usable reading can be trusted: grounded in the page text first, then the
    confidence (an ungrounded value is always low confidence)."""
    return c.grounded is not False, CONFIDENCE_RANK[c.confidence]


def _keeping(winner: Candidate, loser: Candidate) -> Candidate:
    """The winner, with the losing reading kept as an alternative when it says something else."""
    if not loser.usable or _same_value_key(loser.value) == _same_value_key(winner.value):
        return winner
    return replace(winner, alternatives=(*winner.alternatives, loser, *loser.alternatives))


def merge_escalated(first: Candidate | None, second: Candidate | None) -> Candidate | None:
    """Combine a first reading with a stronger engine's re-reading of the same page.

    Escalation re-reads a whole page because one field on it was doubtful, so the other fields
    on that page are read again too. The stronger engine is preferred, but it can be wrong in a
    new way: it can miss a value, and it can invent one. So:
      * second valid, first not valid             -> take the second
      * both valid, second at least as trusted    -> take the second (first kept as alternative)
      * both valid, second less trusted           -> keep the first (second kept as alternative)
      * second reading invalid, first valid       -> keep the first
      * second says unreadable                    -> unreadable, first reading as the partial
      * second found nothing, first was confident -> keep the first
      * second found nothing, first saw something -> unreadable with the first as partial: ask
    "Trusted" means grounded in the page text, then confidence: a valid value that cannot be
    traced to the page never replaces one that can.
    """
    if second is None:
        return first
    second = replace(second, escalated=True)
    if first is None:
        return second
    if second.usable:
        if not first.usable or _trust(second) >= _trust(first):
            return _keeping(second, first)
        return _keeping(replace(first, escalated=True), second)
    if second.value is not None:  # the stronger reading is invalid
        return replace(first, escalated=True) if first.usable else second
    if second.unreadable:
        return replace(second, partial=second.partial or _partial_of(first))
    if first.trusted:
        return replace(first, escalated=True)
    if first.seen:
        return Candidate(field=first.field, page=first.page, source=second.source, value=None,
                         raw=first.raw, confidence="none", evidence=first.evidence,
                         unreadable=True, partial=_partial_of(first), escalated=True)
    return second


@dataclass
class FieldResult:
    name: str
    status: str
    value: Any = None
    confidence: str = "none"
    page: int | None = None
    evidence: str | None = None
    source: str | None = None
    grounded: bool | None = None
    partial: str | None = None
    raw: Any = None
    issues: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    pages_unread: list[int] = field(default_factory=list)
    escalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "value": self.value,
                               "confidence": self.confidence, "page": self.page,
                               "evidence": self.evidence, "source": self.source}
        if self.grounded is not None:
            out["grounded"] = self.grounded
        if self.partial:
            out["partial"] = self.partial
        if self.status == "invalid":
            out["raw"] = self.raw
            out["issues"] = self.issues
        if self.alternatives:
            out["alternatives"] = self.alternatives
        if self.pages_unread:
            out["pages_unread"] = self.pages_unread
        if self.escalated:
            out["escalated"] = True
        return out


def _same_value_key(value: Any) -> str:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.2f}"
    return json.dumps(value, sort_keys=True, default=str)


def _page_order(c: Candidate) -> int:
    return c.page if c.page is not None else 10**6


def resolve_field(spec: FieldSpec, candidates: Collection[Candidate | None],
                  unread_pages: Collection[int]) -> FieldResult:
    cands = [c for c in candidates if c is not None]
    unread = sorted(set(unread_pages))
    usable = [c for c in cands if c.usable]
    if usable:
        best = max(usable, key=lambda c: (CONFIDENCE_RANK[c.confidence], c.grounded is not False,
                                          -_page_order(c)))
        seen = {_same_value_key(best.value)}
        alternatives = []
        readings = usable + [alt for c in usable for alt in c.alternatives if alt.usable]
        for c in sorted(readings, key=_page_order):
            key = _same_value_key(c.value)
            if key not in seen:
                seen.add(key)
                alternatives.append({"value": c.value, "page": c.page,
                                     "confidence": c.confidence})
        return FieldResult(spec.name, "found", value=best.value, confidence=best.confidence,
                           page=best.page, evidence=best.evidence, source=best.source,
                           grounded=best.grounded, alternatives=alternatives,
                           escalated=best.escalated)
    unreadable = [c for c in cands if c.unreadable]
    if unreadable:
        c = min(unreadable, key=lambda c: (c.partial is None, _page_order(c)))
        return FieldResult(spec.name, "unreadable", page=c.page, evidence=c.evidence,
                           source=c.source, partial=c.partial, escalated=c.escalated)
    invalid = [c for c in cands if c.value is not None]
    if invalid:
        c = max(invalid, key=lambda c: (CONFIDENCE_RANK[c.confidence], -_page_order(c)))
        return FieldResult(spec.name, "invalid", page=c.page, evidence=c.evidence,
                           source=c.source, raw=c.raw, issues=list(c.issues),
                           escalated=c.escalated)
    return FieldResult(spec.name, "not_checked" if unread else "not_found", pages_unread=unread)
