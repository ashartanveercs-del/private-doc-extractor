"""Is a returned value actually written on the page?

A model can produce a well-formed, schema-valid value that is nowhere in the document. The
check is deliberately simple and explainable:

* the cited evidence must occur verbatim in the page text (ignoring case, spacing and
  punctuation), starting and ending on word boundaries, and
* the value must be consistent with that evidence: the same words for text, the same amount
  for numbers ("$1,250.00" grounds 1250.0), the same calendar day for dates.

Text is matched word by word, never as a bare substring: "12" is not on a page that only says
"INV-41234", "Acme" is not "Acmeco", and a value with no letters or digits at all is never
grounded. Without evidence, the value itself is looked up in the page text. Lists and objects
are grounded only if every value inside them is. An ungrounded value is not thrown away: its
confidence drops to "low", which flags it for escalation and for a human.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from .coerce import find_dates, find_numbers
from .schema import primary_type


def normalize_text(text: str | None) -> str:
    return "".join(ch for ch in (text or "").casefold() if ch.isalnum())


@functools.lru_cache(maxsize=256)
def _words(text: str) -> tuple[str, frozenset[int], frozenset[int]]:
    """The text's letters and digits as one string, plus the offsets where its words start and
    end. A word is a run of letters and digits; a comma or period between two digits stays
    inside its word, so "$1,250.00" is the single word "125000" and "1" is not one of its words.
    """
    folded = text.casefold()
    joined: list[str] = []
    starts: set[int] = set()
    ends: set[int] = set()
    in_word = False
    for i, ch in enumerate(folded):
        if ch.isalnum():
            if not in_word:
                starts.add(len(joined))
                in_word = True
            joined.append(ch)
        elif (ch in ",." and in_word and joined[-1].isdigit()
              and folded[i + 1:i + 2].isdigit()):
            continue  # a thousands separator or decimal point
        elif in_word:
            ends.add(len(joined))
            in_word = False
    if in_word:
        ends.add(len(joined))
    return "".join(joined), frozenset(starts), frozenset(ends)


def contains_words(text: str | None, phrase: str | None) -> bool:
    """True when `phrase` occurs in `text` as whole words.

    Case, spacing and punctuation are ignored ("INV-1042" matches "inv 1042"), but a match must
    start where a word starts and end where a word ends. An empty phrase never matches.
    """
    wanted = normalize_text(phrase)
    if not wanted:
        return False
    joined, starts, ends = _words(text or "")
    at = joined.find(wanted)
    while at >= 0:
        if at in starts and at + len(wanted) in ends:
            return True
        at = joined.find(wanted, at + 1)
    return False


def is_grounded(value: Any, evidence: str | None, page_text: str | None,
                node: Mapping[str, Any], date_order: str = "MDY") -> bool:
    if value is None or not normalize_text(page_text):
        return False
    if evidence:
        if not contains_words(page_text, evidence):
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
        if node.get("format") == "date" and value in find_dates(source, date_order):
            return True
        return contains_words(source, value)
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
