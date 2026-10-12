import json

import pytest

from privdoc.errors import SchemaError
from privdoc.schema import BUILTIN_SCHEMAS, DocSchema, load_schema

VALIDATION_ONLY = {"pattern", "format", "minimum", "maximum", "exclusiveMinimum",
                   "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems", "const"}


def small(**props):
    return {"type": "object", "properties": props or {"a": {"type": "string"}}}


def test_builtin_schemas_load_by_name():
    assert set(BUILTIN_SCHEMAS) == {"invoice", "receipt", "tax_form"}
    invoice = load_schema("invoice")
    assert [f.name for f in invoice.fields][:3] == ["vendor_name", "invoice_number", "invoice_date"]
    assert invoice.field("total").required and not invoice.field("po_number").required
    assert invoice.field("invoice_date").format == "date"
    assert invoice.field("line_items").type == "array"


def test_schema_loads_from_a_path_or_inline_json(tmp_path):
    path = tmp_path / "ticket.json"
    path.write_text(json.dumps(small(subject={"type": "string"})), encoding="utf-8")
    assert load_schema(str(path)).field("subject").type == "string"
    assert load_schema('{"type": "object", "properties": {"x": {"type": "integer"}}}').field(
        "x").type == "integer"


@pytest.mark.parametrize("bad, where", [
    ({"oneOf": [{"type": "string"}, {"type": "number"}]}, "/properties/x"),
    ({"$ref": "#/$defs/thing"}, "/properties/x"),
    ({"type": "object", "properties": {"y": {"type": "string"}},
      "patternProperties": {"^z": {"type": "string"}}}, "/properties/x"),
    ({"type": "string", "format": "uri"}, "/properties/x"),
])
def test_keywords_that_cannot_be_enforced_are_refused_with_their_location(bad, where):
    with pytest.raises(SchemaError) as info:
        DocSchema(small(x=bad))
    assert where in str(info.value)


@pytest.mark.parametrize("raw, message", [
    ({"type": "array", "items": {"type": "string"}}, "object"),
    ({"type": "object"}, "properties"),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]}, "b"),
    ({"type": "object", "properties": {"total amount": {"type": "number"}}}, "total amount"),
    ({"type": "object", "properties": {"a": {"type": "string"}},
      "additionalProperties": True}, "additionalProperties"),
    (small(a={"type": ["string", "number"]}), "one type"),
    (small(a={"type": "array"}), "items"),
    (small(a={"type": "string", "pattern": "(unclosed"}), "pattern"),
    (small(a={"type": "string", "x-labels": "Total"}), "x-labels"),
    (small(a={"type": "string", "enum": ["x", 3]}), "enum"),
    (small(a={"type": "date"}), "type"),
])
def test_structural_problems_are_reported(raw, message):
    with pytest.raises(SchemaError, match=message):
        DocSchema(raw)


def test_too_many_fields_is_refused():
    with pytest.raises(SchemaError, match="at most"):
        DocSchema({"type": "object",
                   "properties": {f"f{i}": {"type": "string"} for i in range(41)}})


def test_annotations_and_x_extensions_are_accepted():
    schema = DocSchema(small(a={"type": "string", "title": "Account", "description": "d",
                                "examples": ["x"], "x-ui-hint": "wide"}))
    assert schema.field("a").description == "d"


def test_labels_come_from_x_labels_then_title_then_the_name():
    schema = DocSchema(small(invoice_number={"type": "string", "title": "Invoice no.",
                                             "x-labels": ["Invoice #", "invoice NO."]}))
    assert schema.field("invoice_number").labels == ("Invoice #", "invoice NO.",
                                                     "invoice number")


@pytest.fixture
def invoice():
    return load_schema("invoice")


def good_invoice():
    return {"vendor_name": "Cedar Lane Plumbing", "invoice_number": "INV-1042",
            "invoice_date": "2026-03-14", "total": 1250.0,
            "line_items": [{"description": "Pipe fitting kit", "quantity": 2, "unit_price": 12.5,
                            "amount": 25.0}]}


def test_valid_data_has_no_issues(invoice):
    assert invoice.validate(good_invoice()) == []


@pytest.mark.parametrize("change, path, message", [
    ({"total": "1,250.00"}, "/total", "expected number"),
    ({"total": True}, "/total", "expected number"),
    ({"total": -5}, "/total", "minimum"),
    ({"invoice_date": "2026-02-30"}, "/invoice_date", "date"),
    ({"invoice_date": "03/14/2026"}, "/invoice_date", "date"),
    ({"line_items": [{"description": "x", "quantity": 1, "unit_price": 1, "amount": "one"}]},
     "/line_items/0/amount", "expected number"),
    ({"line_items": [{"description": "x", "quantity": 1, "unit_price": 1}]},
     "/line_items/0/amount", "required"),
    ({"line_items": [{"description": "x", "quantity": 1, "unit_price": 1, "amount": 1,
                      "sku": "A1"}]}, "/line_items/0/sku", "not allowed"),
    ({"surprise": 1}, "/surprise", "not allowed"),
])
def test_violations_are_reported_with_precise_paths(invoice, change, path, message):
    issues = invoice.validate({**good_invoice(), **change})
    assert any(i.path == path and message in i.message for i in issues), issues


def test_missing_required_fields_are_reported(invoice):
    data = good_invoice()
    del data["total"]
    assert [str(i) for i in invoice.validate(data)] == ["/total: required field is missing"]


def test_other_keywords_are_enforced():
    schema = DocSchema(small(
        code={"type": "string", "pattern": "^[A-Z]{2}$", "minLength": 2},
        n={"type": "integer", "maximum": 10, "exclusiveMinimum": 0},
        kind={"type": "string", "enum": ["a", "b"]},
        flag={"type": "boolean", "const": True},
        tags={"type": "array", "items": {"type": "string"}, "maxItems": 1},
        email={"type": "string", "format": "email"},
    ))
    issues = schema.validate({"code": "abc", "n": 0, "kind": "c", "flag": False,
                              "tags": ["x", "y"], "email": "nobody"})
    assert {i.path for i in issues} == {"/code", "/n", "/kind", "/flag", "/tags", "/email"}
    assert schema.validate({"code": "AB", "n": 3.0, "kind": "a", "flag": True, "tags": ["x"],
                            "email": "billing@example.com"}) == []


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


@pytest.mark.parametrize("which", ["page", "document"])
def test_model_schemas_are_strict_everywhere(invoice, which):
    model = invoice.page_schema() if which == "page" else invoice.document_schema()
    json.dumps(model)
    for node in _walk(model):
        if node.get("type") == "object" or "object" in (node.get("type") or []):
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
        assert not VALIDATION_ONLY & set(node), node
    fields = model["properties"]["fields"]["properties"]
    assert set(fields) == {f.name for f in invoice.fields}
    wrapper = fields["total"]["properties"]
    expected = {"value", "unreadable", "confidence", "evidence"} | (
        {"page"} if which == "document" else set())
    assert set(wrapper) == expected
    assert wrapper["value"]["type"] == ["number", "null"]
    assert "minimum 0" in wrapper["value"]["description"]


def test_page_schema_asks_for_a_transcription_and_legibility(invoice):
    props = invoice.page_schema()["properties"]
    assert props["transcription"]["type"] == "string"
    assert props["legibility"]["enum"] == ["good", "fair", "poor"]


def test_fingerprint_ignores_key_order_but_tracks_content():
    a = DocSchema({"type": "object", "properties": {"x": {"type": "string"},
                                                    "y": {"type": "number"}}})
    b = DocSchema({"properties": {"y": {"type": "number"}, "x": {"type": "string"}},
                   "type": "object"})
    c = DocSchema({"type": "object", "properties": {"x": {"type": "string"},
                                                    "y": {"type": "integer"}}})
    assert a.fingerprint == b.fingerprint != c.fingerprint


def test_field_guide_lists_names_types_constraints_and_labels(invoice):
    guide = invoice.field_guide()
    assert "- invoice_date (date, YYYY-MM-DD, required)" in guide
    assert 'labels: "Invoice date", "Date"' in guide
    assert "line_items (list of objects" in guide
