"""Deterministic repair: bring a model's (or a regex's) raw value to the schema's type.

This runs before any model is asked to fix anything, because most "invalid" outputs are
formatting, not misreadings: "$1,250.00" for a number, "03/14/2026" for a date, "Yes" for a
boolean. Rules are conservative: anything ambiguous is left alone for the validator to reject,
rather than guessed into a plausible wrong value.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping
from typing import Any

from .schema import primary_type

NULL_WORDS = frozenset({"", "null", "none", "n/a", "na", "-", "--", "–", "—"})
_TRUE = frozenset({"yes", "y", "true", "t", "1", "x", "[x]", "(x)", "☒", "☑",
                   "✓", "✔", "checked", "on"})
_FALSE = frozenset({"no", "n", "false", "f", "0", "[ ]", "[]", "( )", "☐", "unchecked",
                    "off"})

_CURRENCY = re.compile(r"(?i)\b(?:usd|eur|gbp|cad|aud|chf|jpy|inr)\b|[$€£¥₹]")
_US_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_EU_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?|\d{1,3}(?:\.\d{3})+,\d+")
_DECIMAL_COMMA = re.compile(r"\d+,\d{1,2}")
_PLAIN = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.,])\(?-?(?:[$€£¥₹]\s?)?\d(?:[\d,.]*\d)?\)?")

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = (r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
          r"|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_BEFORE = r"(?<![\w/.\-])"
_AFTER = r"(?![\w/\-]|\.\d)"
_DATE_PATTERNS = [
    re.compile(_BEFORE + r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})" + _AFTER),
    re.compile(_BEFORE + r"(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<y>\d{4}|\d{2})" + _AFTER),
    re.compile(_BEFORE + _MONTH + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>\d{4})" + _AFTER,
               re.IGNORECASE),
    re.compile(_BEFORE + r"(?P<d>\d{1,2})(?:st|nd|rd|th)?[\s-]+" + _MONTH + r"[\s,-]+(?P<y>\d{4})"
               + _AFTER, re.IGNORECASE),
]


def parse_number(text: Any) -> float | None:
    """'$1,250.00' -> 1250.0, '1.250,00' -> 1250.0, '(12.50)' -> -12.5; None if unsure."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    s = _CURRENCY.sub("", s).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if s.endswith("-"):
        negative, s = True, s[:-1].strip()
    if s.startswith("-"):
        negative, s = True, s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    s = s.replace(" ", "").replace(" ", "").replace(" ", "")
    if _US_THOUSANDS.fullmatch(s):
        s = s.replace(",", "")
    elif _EU_THOUSANDS.fullmatch(s):
        s = s.replace(".", "").replace(",", ".")
    elif _DECIMAL_COMMA.fullmatch(s):
        s = s.replace(",", ".")
    elif not _PLAIN.fullmatch(s):
        return None
    value = float(s)
    return -value if negative else value


def find_numbers(text: str) -> list[float]:
    out = []
    for match in _NUMBER_TOKEN.finditer(text or ""):
        value = parse_number(match.group(0))
        if value is not None:
            out.append(value)
    return out


def _year(raw: str) -> int:
    year = int(raw)
    if len(raw) == 2:
        year += 2000 if year < 70 else 1900
    return year


def _from_match(match: re.Match[str], order: str) -> str | None:
    groups = match.groupdict()
    try:
        year = _year(groups["y"])
        if groups.get("mon"):
            month = _MONTHS[groups["mon"].lower()[:3]]
            day = int(groups["d"])
        elif groups.get("a"):
            a, b = int(groups["a"]), int(groups["b"])
            if a > 12 >= b:
                day, month = a, b
            elif b > 12 >= a:
                month, day = a, b
            elif order.upper() == "DMY":
                day, month = a, b
            else:
                month, day = a, b
        else:
            month, day = int(groups["m"]), int(groups["d"])
        return dt.date(year, month, day).isoformat()
    except (ValueError, KeyError):
        return None


def parse_date(text: Any, order: str = "MDY") -> str | None:
    """A single date in a common format -> 'YYYY-MM-DD'. `order` settles 04/03 vs 03/04."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    for pattern in _DATE_PATTERNS:
        match = pattern.fullmatch(s)
        if match:
            return _from_match(match, order)
    return None


def find_dates(text: str, order: str = "MDY") -> list[str]:
    hits: list[tuple[int, int, str]] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text or ""):
            iso = _from_match(match, order)
            if iso:
                hits.append((match.start(), -(match.end() - match.start()), iso))
    out, covered = [], -1
    for start, negative_len, iso in sorted(hits):
        if start >= covered:
            out.append(iso)
            covered = start - negative_len
    return out


def parse_bool(text: Any) -> bool | None:
    if not isinstance(text, str):
        return None
    s = " ".join(text.lower().split())
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _match_enum(options: list[Any], value: str) -> Any:
    key = value.casefold()
    for option in options:
        if isinstance(option, str) and option.casefold() == key:
            return option
    return value


def coerce(node: Mapping[str, Any], value: Any, date_order: str = "MDY") -> Any:
    if value is None:
        return None
    kind = primary_type(node)
    if isinstance(value, str):
        value = " ".join(value.split())
        if value.lower() in NULL_WORDS and value not in (node.get("enum") or ()):
            return None
    if kind == "string":
        if isinstance(value, int | float) and not isinstance(value, bool):
            value = _format_number(value)
        if isinstance(value, str):
            if "enum" in node:
                return _match_enum(node["enum"], value)
            if node.get("format") == "date":
                return parse_date(value, date_order) or value
        return value
    if kind in ("number", "integer"):
        if isinstance(value, str):
            parsed = parse_number(value)
            value = value if parsed is None else parsed
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        return value
    if kind == "boolean":
        if isinstance(value, str):
            parsed_bool = parse_bool(value)
            return value if parsed_bool is None else parsed_bool
        if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        return value
    if kind == "array":
        items = value if isinstance(value, list) else [value]
        return [coerce(node["items"], item, date_order) for item in items]
    if kind == "object" and isinstance(value, Mapping):
        props = node["properties"]
        return {key: coerce(props[key], item, date_order) for key, item in value.items()
                if key in props}
    return value


_FENCE = re.compile(r"^```[\w-]*[ \t]*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)


def parse_json_object(text: Any) -> dict[str, Any]:
    """Read a JSON object from model output, tolerating code fences and surrounding prose."""
    if not isinstance(text, str):
        raise ValueError("model output is not text")
    s = text.strip()
    fence = _FENCE.match(s)
    if fence:
        s = fence.group("body").strip()
    try:
        obj = json.loads(s)
    except ValueError:
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the model output") from None
        obj = json.loads(s[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is JSON but not an object")
    return obj
