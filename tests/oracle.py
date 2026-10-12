"""A perfect "model" built from synthetic ground truth. Tests perturb its answers on purpose."""

from __future__ import annotations

import re
from typing import Any

from privdoc.grounding import is_grounded
from privdoc.schema import load_schema
from privdoc.synth import SynthDoc, variant_pages
from tests.fakes import fval


def page_texts(doc: SynthDoc, variant: str) -> list[str]:
    return ["\n".join(line.text for line in lines) for lines in variant_pages(doc, variant)]


def evidence_for(text: str, labels) -> str | None:
    lines = text.splitlines()
    for label in labels:
        rx = re.compile(r"(?<!\w)" + r"\s*".join(map(re.escape, label.split())) + r"(?!\w)",
                        re.IGNORECASE)
        for line in lines:
            if rx.search(line):
                return line.strip()
    return None


def _on_page(doc: SynthDoc, text: str, name: str) -> dict[str, Any] | None:
    """The truth value with a verbatim evidence line, if the value is on this page."""
    spec = load_schema(doc.kind).field(name)
    value = doc.truth.get(name)
    if value is None:
        return None
    candidates = [evidence_for(text, spec.labels)] + [line.strip() for line in text.splitlines()]
    for evidence in candidates:
        if evidence and is_grounded(value, evidence, text, spec.schema):
            return fval(value, evidence=evidence)
    return None


def oracle_page(doc: SynthDoc, variant: str, page: int, **overrides) -> dict[str, Any]:
    """A vision model's answer for one page."""
    text = page_texts(doc, variant)[page - 1]
    fields = {}
    for name in doc.fields:
        fields[name] = _on_page(doc, text, name) or fval(None)
    fields.update(overrides)
    return {"transcription": text, "legibility": "good", "fields": fields}


def oracle_fields(doc: SynthDoc, variant: str, pages: list[int], **overrides) -> dict[str, Any]:
    """A text model's answer for the given pages."""
    texts = page_texts(doc, variant)
    fields = {}
    for name in doc.fields:
        fields[name] = fval(None)
        for number in pages:
            found = _on_page(doc, texts[number - 1], name)
            if found:
                fields[name] = {**found, "page": number}
                break
    fields.update(overrides)
    return {"fields": fields}
