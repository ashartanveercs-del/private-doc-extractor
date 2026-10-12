"""A strict, fully enforced subset of JSON Schema.

"Schema-strict" output means two things here:

1. The schema itself is checked when it is loaded. A keyword this module cannot enforce
   (``oneOf``, ``$ref``, ``patternProperties``, an unknown ``format`` ...) is refused with its
   location instead of being silently ignored, because a validator that skips half the schema
   is worse than none. Objects are always closed (``additionalProperties: false``) and every
   schema node needs one concrete ``type`` (optionally alongside ``null``).
2. Model-facing schemas are generated from it in the shape structured-output APIs require in
   strict mode: every property listed in ``required``, optional values made nullable, and
   validation-only keywords (``pattern``, ``minimum``, ``format`` ...) moved into descriptions.
   Those constraints are then enforced on the way back by :meth:`DocSchema.validate`.

Each top-level property is one *field*. Models answer per field with a wrapper::

    {"value": <the field's own schema, nullable>, "unreadable": bool,
     "confidence": "high"|"medium"|"low"|"none", "evidence": str|null, ["page": int|null]}

so "not on the page", "on the page but unreadable" and "read with low confidence" stay
distinguishable all the way to the output.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .errors import SchemaError

ANNOTATIONS = frozenset({"$schema", "$id", "$comment", "title", "description", "examples",
                         "default", "deprecated", "readOnly", "writeOnly"})
KEYWORD_TYPES: dict[str, frozenset[str]] = {
    "type": frozenset(), "enum": frozenset(), "const": frozenset(),
    "properties": frozenset({"object"}), "required": frozenset({"object"}),
    "additionalProperties": frozenset({"object"}), "items": frozenset({"array"}),
    "minItems": frozenset({"array"}), "maxItems": frozenset({"array"}),
    "minLength": frozenset({"string"}), "maxLength": frozenset({"string"}),
    "pattern": frozenset({"string"}), "format": frozenset({"string"}),
    "minimum": frozenset({"number", "integer"}), "maximum": frozenset({"number", "integer"}),
    "exclusiveMinimum": frozenset({"number", "integer"}),
    "exclusiveMaximum": frozenset({"number", "integer"}),
}
TYPES = frozenset({"string", "number", "integer", "boolean", "null", "array", "object"})
FORMATS = {"date": "YYYY-MM-DD", "date-time": "ISO 8601", "email": "an email address"}
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
MAX_FIELDS = 40
MAX_DEPTH = 4
CONFIDENCES = ("high", "medium", "low", "none")
LEGIBILITY = ("good", "fair", "poor")

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_BUILTIN_DIR = "schemas"
BUILTIN_SCHEMAS = ("invoice", "receipt", "tax_form")


def type_list(node: Mapping[str, Any]) -> list[str]:
    declared = node.get("type")
    return [declared] if isinstance(declared, str) else list(declared or [])


def primary_type(node: Mapping[str, Any]) -> str:
    """The one non-null type of a (checked) schema node."""
    return next(t for t in type_list(node) if t != "null")


def allows_null(node: Mapping[str, Any]) -> bool:
    return "null" in type_list(node)


@dataclass(frozen=True)
class Issue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    schema: Mapping[str, Any]
    required: bool
    type: str
    nullable: bool
    format: str | None
    labels: tuple[str, ...]
    description: str


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def labels_for(name: str, node: Mapping[str, Any]) -> tuple[str, ...]:
    """Printed labels a value may appear under: x-labels, then the title, then the name."""
    candidates = list(node.get("x-labels") or [])
    if node.get("title"):
        candidates.append(node["title"])
    candidates.append(name.replace("_", " "))
    seen, out = set(), []
    for label in candidates:
        key = " ".join(label.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(label)
    return tuple(out)


class DocSchema:
    def __init__(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise SchemaError("/: the schema must be a JSON object")
        _check(raw, "", 0)
        if primary_type(raw) != "object" or not raw.get("properties"):
            raise SchemaError("/: the top level must be an object with properties")
        if len(raw["properties"]) > MAX_FIELDS:
            raise SchemaError(f"/: at most {MAX_FIELDS} fields are supported")
        self.raw = raw
        required = set(raw.get("required") or [])
        self.fields = [
            FieldSpec(name=name, schema=node, required=name in required,
                      type=primary_type(node), nullable=allows_null(node),
                      format=node.get("format"), labels=labels_for(name, node),
                      description=str(node.get("description") or ""))
            for name, node in raw["properties"].items()]
        self._by_name = {f.name: f for f in self.fields}
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.title = str(raw.get("title") or "document")

    def field(self, name: str) -> FieldSpec:
        return self._by_name[name]

    # -- validation ----------------------------------------------------------------------------

    def validate(self, data: Any) -> list[Issue]:
        issues: list[Issue] = []
        if not isinstance(data, Mapping):
            return [Issue("/", "expected an object")]
        for spec in self.fields:
            if spec.name not in data:
                if spec.required:
                    issues.append(Issue(f"/{spec.name}", "required field is missing"))
                continue
            _validate(spec.schema, data[spec.name], f"/{spec.name}", issues)
        for key in data:
            if key not in self._by_name:
                issues.append(Issue(f"/{key}", "property not allowed by the schema"))
        return issues

    def validate_field(self, name: str, value: Any) -> list[Issue]:
        issues: list[Issue] = []
        _validate(self.field(name).schema, value, f"/{name}", issues)
        return issues

    # -- model-facing schemas ------------------------------------------------------------------

    def page_schema(self) -> dict[str, Any]:
        """For a vision model reading one page image."""
        return _closed({
            "transcription": {"type": "string", "description": (
                "Every piece of text on the page, verbatim, in reading order. Write [illegible] "
                "for characters you cannot read.")},
            "legibility": {"type": "string", "enum": list(LEGIBILITY)},
            "fields": self._fields_object(with_page=False),
        })

    def document_schema(self) -> dict[str, Any]:
        """For a text model reading the text of one or more pages."""
        return _closed({"fields": self._fields_object(with_page=True)})

    def _fields_object(self, with_page: bool) -> dict[str, Any]:
        return _closed({f.name: _wrapper(f, with_page) for f in self.fields})

    def field_guide(self) -> str:
        lines = []
        for f in self.fields:
            kind = _kind(f.schema) + (", required" if f.required else "")
            line = f"- {f.name} ({kind})"
            if f.description:
                line += f": {f.description}"
            shown = [label for label in f.labels if label != f.name.replace("_", " ")]
            if shown:
                line += " labels: " + ", ".join(f'"{label}"' for label in shown)
            lines.append(line)
        return "\n".join(lines)


def load_schema(spec: str | Path | Mapping[str, Any]) -> DocSchema:
    """A built-in name ("invoice"), a path to a JSON file, inline JSON, or a dict."""
    if isinstance(spec, Mapping):
        return DocSchema(spec)
    text = str(spec)
    if text in BUILTIN_SCHEMAS:
        source = resources.files("privdoc") / _BUILTIN_DIR / f"{text}.schema.json"
        return DocSchema(json.loads(source.read_text(encoding="utf-8")))
    if text.lstrip().startswith("{"):
        raw_text, where = text, "inline schema"
    else:
        path = Path(text)
        try:
            raw_text, where = path.read_text(encoding="utf-8"), path.name
        except OSError:
            raise SchemaError(f"cannot read schema file {text!r}; built-in schemas are: "
                              + ", ".join(BUILTIN_SCHEMAS)) from None
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise SchemaError(f"{where} is not valid JSON (line {err.lineno})") from None
    return DocSchema(raw)


# -- schema checking ---------------------------------------------------------------------------

def _check(node: Any, path: str, depth: int) -> None:
    where = path or "/"
    if not isinstance(node, Mapping):
        raise SchemaError(f"{where}: a schema must be an object")
    for key in node:
        if key.startswith("x-"):
            continue
        if key not in ANNOTATIONS and key not in KEYWORD_TYPES:
            raise SchemaError(f"{where}: unsupported keyword {key!r}; supported: "
                              + ", ".join(sorted(KEYWORD_TYPES)))
    labels = node.get("x-labels")
    if labels is not None and not (isinstance(labels, list) and labels and all(
            isinstance(x, str) and x.strip() for x in labels)):
        raise SchemaError(f"{where}: x-labels must be a list of non-empty strings")
    types = type_list(node)
    if not types:
        raise SchemaError(f"{where}: every schema needs a 'type'")
    unknown = [t for t in types if t not in TYPES]
    if unknown:
        raise SchemaError(f"{where}: unknown type {unknown[0]!r}")
    concrete = [t for t in types if t != "null"]
    if len(concrete) != 1:
        raise SchemaError(f"{where}: give exactly one type (optionally with null), "
                          f"not {types}")
    kind = concrete[0]
    for key, applies in KEYWORD_TYPES.items():
        if key in node and applies and kind not in applies:
            raise SchemaError(f"{where}: {key!r} does not apply to type {kind!r}")

    if kind == "object":
        if depth >= MAX_DEPTH:
            raise SchemaError(f"{where}: objects nest at most {MAX_DEPTH} levels deep")
        props = node.get("properties")
        if not isinstance(props, Mapping) or not props:
            raise SchemaError(f"{where}: an object needs properties")
        for name, child in props.items():
            if not NAME_RE.fullmatch(str(name)):
                raise SchemaError(f"{where}: property name {name!r} must be letters, digits "
                                  "and underscores, starting with a letter")
            _check(child, f"{path}/properties/{name}", depth + 1)
        required = node.get("required", [])
        if not isinstance(required, list) or any(r not in props for r in required):
            missing = [r for r in required if r not in props] if isinstance(required, list) else []
            raise SchemaError(f"{where}: required lists unknown properties {missing}")
        if node.get("additionalProperties", False) is not False:
            raise SchemaError(f"{where}: additionalProperties must be false (objects are "
                              "closed so extra keys can never slip through)")
    elif kind == "array":
        if not isinstance(node.get("items"), Mapping):
            raise SchemaError(f"{where}: an array needs a single 'items' schema")
        _check(node["items"], f"{path}/items", depth + 1)

    if "enum" in node:
        enum = node["enum"]
        if not isinstance(enum, list) or not enum or any(
                not _is_type(v, kind) and not (v is None and "null" in types) for v in enum):
            raise SchemaError(f"{where}: enum values must all be of type {kind!r}")
    if "pattern" in node:
        try:
            re.compile(node["pattern"])
        except (re.error, TypeError):
            raise SchemaError(f"{where}: invalid pattern {node['pattern']!r}") from None
    if "format" in node and node["format"] not in FORMATS:
        raise SchemaError(f"{where}: unsupported format {node['format']!r}; supported: "
                          + ", ".join(FORMATS))
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in node and not _is_type(node[key], "number"):
            raise SchemaError(f"{where}: {key} must be a number")
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in node and not (isinstance(node[key], int) and not isinstance(node[key], bool)
                                and node[key] >= 0):
            raise SchemaError(f"{where}: {key} must be a non-negative integer")


# -- validation --------------------------------------------------------------------------------

def _is_type(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer())
    if kind == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool)
                and math.isfinite(value))
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, Mapping)
    return False


def _type_name(value: Any) -> str:
    for kind in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_type(value, kind):
            return "number" if kind == "integer" else kind
    return type(value).__name__


def _valid_format(fmt: str, value: str) -> bool:
    if fmt == "date":
        if not _DATE_RE.fullmatch(value):
            return False
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if fmt == "date-time":
        try:
            dt.datetime.fromisoformat(value)
        except ValueError:
            return False
        return "T" in value or " " in value
    return bool(_EMAIL_RE.fullmatch(value))


def _validate(node: Mapping[str, Any], value: Any, path: str, out: list[Issue]) -> None:
    types = type_list(node)
    if not any(_is_type(value, t) for t in types):
        out.append(Issue(path, f"expected {' or '.join(types)}, got {_type_name(value)}"))
        return
    if value is None:
        return
    if "enum" in node and value not in node["enum"]:
        out.append(Issue(path, f"must be one of {node['enum']}"))
    if "const" in node and value != node["const"]:
        out.append(Issue(path, f"must be {node['const']!r}"))
    if isinstance(value, str):
        if "minLength" in node and len(value) < node["minLength"]:
            out.append(Issue(path, f"shorter than {node['minLength']} characters"))
        if "maxLength" in node and len(value) > node["maxLength"]:
            out.append(Issue(path, f"longer than {node['maxLength']} characters"))
        if "pattern" in node and not re.search(node["pattern"], value):
            out.append(Issue(path, f"does not match the pattern {node['pattern']}"))
        fmt = node.get("format")
        if fmt and not _valid_format(fmt, value):
            out.append(Issue(path, f"not a valid {fmt} ({FORMATS[fmt]})"))
    elif _is_type(value, "number"):
        if "minimum" in node and value < node["minimum"]:
            out.append(Issue(path, f"below the minimum {_num(node['minimum'])}"))
        if "maximum" in node and value > node["maximum"]:
            out.append(Issue(path, f"above the maximum {_num(node['maximum'])}"))
        if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
            out.append(Issue(path, f"must be greater than {_num(node['exclusiveMinimum'])}"))
        if "exclusiveMaximum" in node and value >= node["exclusiveMaximum"]:
            out.append(Issue(path, f"must be less than {_num(node['exclusiveMaximum'])}"))
    elif isinstance(value, list):
        if "minItems" in node and len(value) < node["minItems"]:
            out.append(Issue(path, f"fewer than {node['minItems']} items"))
        if "maxItems" in node and len(value) > node["maxItems"]:
            out.append(Issue(path, f"more than {node['maxItems']} items"))
        for i, item in enumerate(value):
            _validate(node["items"], item, f"{path}/{i}", out)
    elif isinstance(value, Mapping):
        props = node["properties"]
        for key in node.get("required", []):
            if key not in value:
                out.append(Issue(f"{path}/{key}", "required property is missing"))
        for key, item in value.items():
            if key in props:
                _validate(props[key], item, f"{path}/{key}", out)
            else:
                out.append(Issue(f"{path}/{key}", "property not allowed by the schema"))


# -- strict model schemas ----------------------------------------------------------------------

def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _constraints(node: Mapping[str, Any]) -> list[str]:
    out = []
    if node.get("format"):
        out.append(f"format {node['format']} ({FORMATS[node['format']]})")
    if "pattern" in node:
        out.append(f"must match /{node['pattern']}/")
    for key, words in (("minimum", "minimum"), ("maximum", "maximum"),
                       ("exclusiveMinimum", "greater than"), ("exclusiveMaximum", "less than")):
        if key in node:
            out.append(f"{words} {_num(node[key])}")
    for key, words in (("minLength", "at least {} characters"),
                       ("maxLength", "at most {} characters"),
                       ("minItems", "at least {} items"), ("maxItems", "at most {} items")):
        if key in node:
            out.append(words.format(node[key]))
    if "const" in node:
        out.append(f"always {node['const']!r}")
    return out


def _strict(node: Mapping[str, Any], nullable: bool) -> dict[str, Any]:
    kind = primary_type(node)
    nullable = nullable or allows_null(node)
    if kind == "object":
        required = set(node.get("required", []))
        out: dict[str, Any] = _closed({
            name: _strict(child, nullable=name not in required)
            for name, child in node["properties"].items()})
    elif kind == "array":
        out = {"type": "array", "items": _strict(node["items"], nullable=False)}
    else:
        out = {"type": kind}
        if "enum" in node:
            out["enum"] = [v for v in node["enum"] if v is not None]
    if nullable:
        out["type"] = [out["type"], "null"]
        if "enum" in out:
            out["enum"].append(None)
    description = "; ".join(filter(None, [str(node.get("description") or "")]
                                   + _constraints(node)))
    if description:
        out["description"] = description
    return out


def _wrapper(spec: FieldSpec, with_page: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "value": _strict(spec.schema, nullable=True),
        "unreadable": {"type": "boolean", "description": (
            "true when the field is on the page but cannot be read with certainty; "
            "value must then be null")},
        "confidence": {"type": "string", "enum": list(CONFIDENCES)},
        "evidence": {"type": ["string", "null"], "description": (
            "the exact text from the page that contains the value, copied verbatim")},
    }
    if with_page:
        props["page"] = {"type": ["integer", "null"],
                         "description": "number of the page the value was read from"}
    return _closed(props)


def _kind(node: Mapping[str, Any]) -> str:
    kind = primary_type(node)
    if "enum" in node:
        return "one of: " + ", ".join(str(v) for v in node["enum"] if v is not None)
    if kind == "string":
        fmt = node.get("format")
        text = f"{fmt}, {FORMATS[fmt]}" if fmt else "text"
        if "pattern" in node:
            text += f" matching {node['pattern']}"
        return text
    if kind == "array":
        items = node["items"]
        if primary_type(items) == "object":
            return "list of objects with " + ", ".join(items["properties"])
        return f"list of {_kind(items)}"
    if kind == "object":
        return "object with " + ", ".join(node["properties"])
    return {"integer": "whole number", "boolean": "true/false"}.get(kind, kind)
