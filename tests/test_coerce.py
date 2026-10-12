import pytest

from privdoc.coerce import (
    coerce,
    find_dates,
    find_numbers,
    parse_bool,
    parse_date,
    parse_json_object,
    parse_number,
)


@pytest.mark.parametrize("text, expected", [
    ("$1,250.00", 1250.0), ("1250", 1250.0), ("USD 1,250.00", 1250.0), ("1.250,00", 1250.0),
    ("12,50", 12.5), ("(12.50)", -12.5), ("-3", -3.0), ("8.25%", 8.25), ("1 250,00", 1250.0),
    ("€ 99", 99.0), ("0.5", 0.5), ("25.00-", -25.0),
])
def test_numbers_are_read_from_money_and_locale_formats(text, expected):
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["abc", "1,2,3", "", "12.5.6", "$"])
def test_non_numbers_are_not_guessed(text):
    assert parse_number(text) is None


@pytest.mark.parametrize("text, order, expected", [
    ("2026-03-14", "MDY", "2026-03-14"), ("2026/3/4", "MDY", "2026-03-04"),
    ("03/14/2026", "DMY", "2026-03-14"),  # 14 can only be a day, whatever the order
    ("14/03/2026", "MDY", "2026-03-14"),
    ("04/03/2026", "MDY", "2026-04-03"), ("04/03/2026", "DMY", "2026-03-04"),
    ("March 14, 2026", "MDY", "2026-03-14"), ("14 March 2026", "MDY", "2026-03-14"),
    ("Mar 14 2026", "MDY", "2026-03-14"), ("14-Mar-2026", "MDY", "2026-03-14"),
    ("Sept. 3, 2025", "MDY", "2025-09-03"), ("3/4/26", "MDY", "2026-03-04"),
    ("12/31/99", "MDY", "1999-12-31"), ("14.03.2026", "DMY", "2026-03-14"),
])
def test_dates_become_iso(text, order, expected):
    assert parse_date(text, order) == expected


@pytest.mark.parametrize("text", ["2026-02-30", "13/13/2026", "hello", "March 2026", "2026"])
def test_impossible_or_partial_dates_are_not_invented(text):
    assert parse_date(text, "MDY") is None


def test_dates_and_numbers_are_found_inside_text():
    assert find_dates("Issued 03/14/2026, due April 13, 2026.", "MDY") == [
        "2026-03-14", "2026-04-13"]
    assert find_numbers("Qty 2 @ $12.50 = $1,025.00") == [2.0, 12.5, 1025.0]


@pytest.mark.parametrize("text, expected", [
    ("yes", True), ("[x]", True), ("☒", True), ("X", True), ("checked", True),
    ("no", False), ("[ ]", False), ("☐", False), ("false", False), ("maybe", None),
])
def test_checkbox_and_yes_no_values(text, expected):
    assert parse_bool(text) is expected


NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}
DATE = {"type": "string", "format": "date"}
TEXT = {"type": "string"}


@pytest.mark.parametrize("schema, raw, expected", [
    (NUMBER, "$1,250.00", 1250.0),
    (NUMBER, "N/A", None),
    (INTEGER, "2025", 2025),
    (INTEGER, "3.0", 3),
    (INTEGER, 3.0, 3),
    (DATE, "03/14/2026", "2026-03-14"),
    (DATE, "soon", "soon"),  # left for the validator to reject
    ({"type": "boolean"}, "Yes", True),
    ({"type": "string", "enum": ["cash", "card"]}, " CARD ", "card"),
    ({"type": "string", "enum": ["cash", "card"]}, "bitcoin", "bitcoin"),
    (TEXT, 1042, "1042"),
    (TEXT, 12.5, "12.5"),
    (TEXT, "  Cedar   Lane \n Plumbing ", "Cedar Lane Plumbing"),
    (TEXT, "", None),
    (TEXT, "n/a", None),
])
def test_values_are_coerced_toward_the_schema(schema, raw, expected):
    assert coerce(schema, raw) == expected


def test_integers_with_a_fraction_are_left_for_the_validator():
    assert coerce(INTEGER, "3.5") == 3.5


def test_nested_objects_drop_unknown_keys_and_coerce_children():
    item = {"type": "object", "properties": {"description": TEXT, "amount": NUMBER}}
    schema = {"type": "array", "items": item}
    assert coerce(schema, [{"description": " Desk lamp ", "amount": "$25.00", "sku": "X"}]) == [
        {"description": "Desk lamp", "amount": 25.0}]
    # a single object where a list was expected is wrapped
    assert coerce(schema, {"description": "Desk lamp", "amount": "25"}) == [
        {"description": "Desk lamp", "amount": 25.0}]


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Here is the result:\n{"a": 1}\nThanks!',
])
def test_model_json_is_read_from_fences_and_surrounding_prose(text):
    assert parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"a": ', ""])
def test_unreadable_model_output_raises(text):
    with pytest.raises(ValueError):
        parse_json_object(text)
