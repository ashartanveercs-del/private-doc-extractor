"""Is a returned value actually written on the page?

A model can produce a well-formed, schema-valid value that is nowhere in the document. The
check is deliberately simple and explainable:

* the cited evidence must occur verbatim in the page text (ignoring case, spacing and
  punctuation), and
* the value must be consistent with that evidence: the same characters for text, the same
  amount for numbers ("$1,250.00" grounds 1250.0), the same calendar day for dates.

Without evidence, the value itself is looked up in the page text. Lists and objects are
grounded only if every value inside them is. An ungrounded value is not thrown away: its
confidence drops to "low", which flags it for escalation and for a human.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .coerce import find_dates, find_numbers
from .schema import primary_type


def normalize_text(text: str | None) -> str:
    return "".join(ch for ch in (text or "").casefold() if ch.isalnum())


def is_grounded(value: Any, evidence: str | None, page_text: str | None,
                node: Mapping[str, Any], date_order: str = "MDY") -> bool:
    page = normalize_text(page_text)
    if value is None or not page:
        return False
    if evidence:
        cited = normalize_text(evidence)
        if not cited or cited not in page:
            return False
        return _supported(value, evidence, page_text or "", node, date_order, has_evidence=True)
    return _supported(value, page_text or "", page_text or "", node, date_order,
                      has_evidence=False)


def _supported(value: Any, source: str, page_text: str, node: Mapping[str, Any],
               date_order: str, has_evidence: bool) -> bool:
    if isinstance(value, bool):
        return has_evidence  # "[x]" vs "true" cannot be compared; the cited line must be real
    if isinstance(value, int | float):
        return any(abs(n - value) < 0.005 for n in find_numbers(source))
    if isinstance(value, str):
        wanted = normalize_text(value)
        if node.get("format") == "date" and value in find_dates(source, date_order):
            return True
        return not wanted or wanted in normalize_text(source)
    kind = primary_type(node)
    if isinstance(value, list) and kind == "array":
        return all(item is None or _supported(item, page_text, page_text, node["items"],
                                              date_order, has_evidence)
                   for item in value)
    if isinstance(value, Mapping) and kind == "object":
        props = node["properties"]
        return all(item is None or _supported(item, page_text, page_text, props[key],
                                              date_order, has_evidence)
                   for key, item in value.items() if key in props)
    return False
