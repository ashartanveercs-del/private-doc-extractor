"""Deterministic field extraction from text: labels, types and simple tables. No model, no network.

Used in the local and own-cloud tiers (and whenever no LLM is configured). It finds a field by
its printed labels (``x-labels``, the title, or the field name), takes the text after the label
up to the next label on the same line (or the next line when the label stands alone), and
parses it by type: dates to ISO, money to numbers, checkboxes to booleans, enums and patterns
by matching. When labels overlap ("Due date" contains "date"), the longest label wins, so the
due date is never read as the invoice date. A label must start its line or be followed by a
colon, so prose like "...after the invoice date" is not mistaken for a field.

Arrays of flat objects (invoice line items) are read as a table: a header line naming at least
two of the item's properties, followed by rows of "text, then one number per numeric column".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..coerce import find_dates, find_numbers, parse_bool, parse_number
from ..records import FieldOutput
from ..schema import DocSchema, FieldSpec, labels_for, primary_type
from .base import PageText

_SEPARATOR = re.compile(r"\s*:\s*")
_STRIP = " \t:#-|"
_NUMBER = r"[-(]?[$€£]?\d[\d,]*(?:\.\d+)?\)?"
_SCALARS = ("string", "number", "integer", "boolean")


def _label_regex(label: str) -> re.Pattern[str]:
    tokens = [re.escape(token) for token in label.split()]
    return re.compile(r"(?<!\w)" + r"\s*".join(tokens) + r"(?!\w)", re.IGNORECASE)


def _absent() -> dict[str, Any]:
    return {"value": None, "unreadable": False, "confidence": "none", "evidence": None,
            "page": None}


class RuleExtractor:
    id = "rules"
    destination = None
    can_repair = False

    def __init__(self, date_order: str = "MDY") -> None:
        self.date_order = date_order
        self.fingerprint = f"rules:{date_order}"

    def extract(self, pages: Sequence[PageText], schema: DocSchema) -> FieldOutput:
        scalars = [f for f in schema.fields if f.type in _SCALARS]
        found = self._labelled(pages, scalars)
        for spec in schema.fields:
            if spec.type == "array":
                table = self._table(pages, spec)
                if table:
                    found[spec.name] = table
        return FieldOutput(fields={f.name: found.get(f.name, _absent()) for f in schema.fields})

    def repair(self, pages, schema, previous, problems):
        raise NotImplementedError("the rule extractor has nothing to repair with")

    # -- labelled scalars ----------------------------------------------------------------------

    def _labelled(self, pages: Sequence[PageText],
                  specs: list[FieldSpec]) -> dict[str, dict[str, Any]]:
        patterns = [(spec, _label_regex(label)) for spec in specs for label in spec.labels]
        found: dict[str, dict[str, Any]] = {}
        for page in pages:
            lines = page.text.splitlines()
            for i, line in enumerate(lines):
                hits = self._hits(line, patterns)
                for j, (_, value_start, spec) in enumerate(hits):
                    if spec.name in found:
                        continue
                    stop = hits[j + 1][0] if j + 1 < len(hits) else len(line)
                    segment = line[value_start:stop].strip(_STRIP)
                    evidence = line.strip()
                    last = j == len(hits) - 1
                    if not segment and last and i + 1 < len(lines) and not self._hits(
                            lines[i + 1], patterns):
                        segment = lines[i + 1].strip(_STRIP)
                        evidence = f"{line.strip()} {lines[i + 1].strip()}"
                    value = self._typed(segment[:200], spec)
                    if value is not None:
                        found[spec.name] = {"value": value, "unreadable": False,
                                            "confidence": "medium", "evidence": evidence[:300],
                                            "page": page.page}
        return found

    @staticmethod
    def _hits(line: str, patterns) -> list[tuple[int, int, FieldSpec]]:
        candidates = []
        for spec, regex in patterns:
            for match in regex.finditer(line):
                separator = _SEPARATOR.match(line, match.end())
                at_start = not line[:match.start()].strip()
                if not (separator or at_start):
                    continue
                value_start = separator.end() if separator else match.end()
                candidates.append((match.start(), value_start, match.end() - match.start(),
                                   spec))
        candidates.sort(key=lambda c: (-c[2], c[0]))  # longest label first
        chosen: list[tuple[int, int, int, FieldSpec]] = []
        for cand in candidates:
            if all(cand[1] <= other[0] or cand[0] >= other[1] for other in chosen):
                chosen.append(cand)
        return [(start, value_start, spec)
                for start, value_start, _, spec in sorted(chosen, key=lambda c: c[0])]

    def _typed(self, segment: str, spec: FieldSpec) -> Any:
        if not segment:
            return None
        node = spec.schema
        if "enum" in node:
            for option in node["enum"]:
                if isinstance(option, str) and re.search(
                        rf"(?<!\w){re.escape(option)}(?!\w)", segment, re.IGNORECASE):
                    return option
            return None
        if spec.type == "string":
            if spec.format == "date":
                dates = find_dates(segment, self.date_order)
                return dates[0] if dates else None
            pattern = node.get("pattern")
            if pattern:
                if re.search(pattern, segment):
                    return segment
                for token in segment.split():
                    token = token.strip(",;()")
                    if re.search(pattern, token):
                        return token
                return None
            return segment
        if spec.type in ("number", "integer"):
            numbers = find_numbers(segment)
            if not numbers:
                return None
            value = numbers[0]
            return int(value) if spec.type == "integer" and value.is_integer() else value
        if spec.type == "boolean":
            value = parse_bool(segment)
            if value is None and segment.split():
                value = parse_bool(segment.split()[0])
            return value
        return None

    # -- tables --------------------------------------------------------------------------------

    def _table(self, pages: Sequence[PageText], spec: FieldSpec) -> dict[str, Any] | None:
        items = spec.schema["items"]
        if primary_type(items) != "object":
            return None
        props: Mapping[str, Any] = items["properties"]
        text_cols = [n for n, node in props.items() if primary_type(node) == "string"]
        num_cols = [n for n, node in props.items() if primary_type(node) in ("number", "integer")]
        if len(text_cols) != 1 or not num_cols or len(text_cols) + len(num_cols) != len(props):
            return None
        header_labels = {n: [_label_regex(label) for label in labels_for(n, node)]
                         for n, node in props.items()}
        row = re.compile(r"^(?P<text>.*?[A-Za-z].*?)" + "".join(
            rf"\s+(?P<n{i}>{_NUMBER})" for i in range(len(num_cols))) + r"\s*$")
        for page in pages:
            lines = page.text.splitlines()
            for i, line in enumerate(lines):
                named = sum(any(rx.search(line) for rx in regexes)
                            for regexes in header_labels.values())
                if named < 2:
                    continue
                rows, used = [], [line.strip()]
                for following in lines[i + 1:]:
                    match = row.match(following.strip())
                    numbers = [parse_number(match.group(f"n{k}")) for k in range(len(num_cols))
                               ] if match else []
                    if not match or any(n is None for n in numbers):
                        break
                    item: dict[str, Any] = {text_cols[0]: match.group("text").strip()}
                    for name, number in zip(num_cols, numbers, strict=True):
                        integral = primary_type(props[name]) == "integer" and number.is_integer()
                        item[name] = int(number) if integral else number
                    rows.append(item)
                    used.append(following.strip())
                if rows:
                    return {"value": rows, "unreadable": False, "confidence": "medium",
                            "evidence": " ".join(used)[:300], "page": page.page}
        return None
