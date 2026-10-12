from privdoc.grounding import is_grounded, normalize_text

PAGE = """CEDAR LANE  PLUMBING,
Invoice #: INV-1042
Invoice date: 03/14/2026
Pipe fitting kit 2 12.50 25.00
Copper elbow 4 3.00 12.00
Total due: $1,250.00
Corrected: [x]
"""
TEXT = {"type": "string"}
NUMBER = {"type": "number"}
DATE = {"type": "string", "format": "date"}
ITEMS = {"type": "array", "items": {"type": "object", "properties": {
    "description": TEXT, "quantity": NUMBER, "amount": NUMBER}}}


def test_normalisation_ignores_case_spacing_and_punctuation():
    assert normalize_text("Cedar Lane  Plumbing,") == "cedarlaneplumbing"


def test_text_values_must_appear_on_the_page():
    assert is_grounded("Cedar Lane Plumbing", None, PAGE, TEXT)
    assert is_grounded("INV-1042", "Invoice #: INV-1042", PAGE, TEXT)
    assert not is_grounded("INV-9999", None, PAGE, TEXT)


def test_evidence_must_be_verbatim_page_text():
    assert not is_grounded("INV-1042", "Invoice number INV-1042 (paid)", PAGE, TEXT)


def test_numbers_match_whatever_their_printed_format():
    assert is_grounded(1250.0, "Total due: $1,250.00", PAGE, NUMBER)
    assert not is_grounded(1205.0, "Total due: $1,250.00", PAGE, NUMBER)


def test_dates_match_across_formats():
    assert is_grounded("2026-03-14", "Invoice date: 03/14/2026", PAGE, DATE)
    assert not is_grounded("2026-03-15", "Invoice date: 03/14/2026", PAGE, DATE)


def test_value_must_be_consistent_with_its_evidence_not_just_the_page():
    # 25.00 is on the page, but not in the line the model cited
    assert not is_grounded(25.0, "Total due: $1,250.00", PAGE, NUMBER)


def test_lists_are_grounded_only_if_every_value_is():
    good = [{"description": "Pipe fitting kit", "quantity": 2, "amount": 25.0},
            {"description": "Copper elbow", "quantity": 4, "amount": 12.0}]
    assert is_grounded(good, None, PAGE, ITEMS)
    bad = good + [{"description": "Gold faucet", "quantity": 1, "amount": 999.0}]
    assert not is_grounded(bad, None, PAGE, ITEMS)


def test_booleans_need_evidence():
    assert is_grounded(True, "Corrected: [x]", PAGE, {"type": "boolean"})
    assert not is_grounded(True, None, PAGE, {"type": "boolean"})


def test_nothing_is_grounded_in_an_empty_page():
    assert not is_grounded("INV-1042", None, "", TEXT)
    assert not is_grounded("INV-1042", "INV-1042", "   ", TEXT)
