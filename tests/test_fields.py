import itertools

import pytest

from privdoc.fields import Candidate, merge_escalated, normalize_candidate, resolve_field
from privdoc.schema import load_schema

SCHEMA = load_schema("invoice")
PAGE = "Cedar Lane Plumbing\nInvoice #: INV-1042\nInvoice date: 03/14/2026\nTotal due: $1,250.00"


def norm(name, raw, page=1, text=PAGE, source="model-a"):
    return normalize_candidate(raw, SCHEMA.field(name), SCHEMA, page=page, page_text=text,
                               source=source)


def raw(value, confidence="high", evidence=None, unreadable=False):
    return {"value": value, "confidence": confidence, "evidence": evidence,
            "unreadable": unreadable}


def test_a_clean_reading_is_kept_as_is():
    c = norm("invoice_number", raw("INV-1042", evidence="Invoice #: INV-1042"))
    assert (c.value, c.confidence, c.grounded, c.issues) == ("INV-1042", "high", True, ())
    assert c.usable and c.seen


def test_unreadable_always_means_null_and_keeps_what_was_made_out_as_partial():
    c = norm("invoice_number", raw("INV-10?2", unreadable=True))
    assert (c.value, c.unreadable, c.partial, c.confidence) == (None, True, "INV-10?2", "none")
    assert c.seen and not c.usable


@pytest.mark.parametrize("value, partial", [("INV-[illegible]42", "INV-[illegible]42"),
                                            ("[illegible]", None), ("???", None)])
def test_illegible_markers_turn_a_value_into_unreadable(value, partial):
    c = norm("invoice_number", raw(value))
    assert c.value is None and c.unreadable and c.partial == partial


def test_a_value_that_is_not_on_the_page_is_downgraded_not_dropped():
    c = norm("invoice_number", raw("INV-9999"))
    assert c.value == "INV-9999" and c.confidence == "low" and c.grounded is False


def test_values_are_coerced_then_validated():
    total = norm("total", raw("$1,250.00", evidence="Total due: $1,250.00"))
    assert total.value == 1250.0 and total.usable
    date = norm("invoice_date", raw("next Tuesday"))
    assert date.issues and not date.usable and date.raw == "next Tuesday"


@pytest.mark.parametrize("reported", [None, {}, raw(None), raw(None, confidence="high")])
def test_absent_fields_have_no_value_and_no_confidence(reported):
    c = norm("po_number", reported)
    assert c.value is None and c.confidence == "none" and not c.unreadable and not c.seen


def test_unknown_confidence_words_count_as_low():
    assert norm("invoice_number", raw("INV-1042", confidence="very high")).confidence == "low"
    assert norm("invoice_number", raw("INV-1042", confidence="none")).confidence == "low"


def test_evidence_is_trimmed():
    c = norm("invoice_number", raw("INV-1042", evidence="Invoice #: INV-1042 " + "x" * 1000))
    assert len(c.evidence) <= 300


def test_long_evidence_is_checked_in_full_and_only_trimmed_in_the_output():
    line = "Notes: " + "see the attached schedule " * 15 + "Invoice #: INV-1042"
    c = norm("invoice_number", raw("INV-1042", evidence=line), text=line)
    assert c.grounded and c.confidence == "high" and len(c.evidence) <= 300


# -- escalation merge --------------------------------------------------------------------------

def cand(value=None, confidence="high", unreadable=False, grounded=True, issues=(), partial=None,
         source="model-a"):
    return Candidate(field="invoice_number", page=1, source=source, value=value, raw=value,
                     confidence="none" if value is None else confidence, evidence=None,
                     unreadable=unreadable, partial=partial,
                     grounded=grounded if value is not None else None, issues=tuple(issues))


def test_a_valid_second_reading_wins():
    merged = merge_escalated(cand("INV-1O42", "low"), cand("INV-1042", source="model-b"))
    assert merged.value == "INV-1042" and merged.source == "model-b" and merged.escalated


def test_a_confident_first_value_survives_when_the_stronger_model_misses_it():
    merged = merge_escalated(cand("INV-1042", "high"), cand(None, source="model-b"))
    assert merged.value == "INV-1042" and merged.source == "model-a"


def test_a_shaky_first_value_becomes_unreadable_when_the_stronger_model_sees_nothing():
    merged = merge_escalated(cand("INV-1O42", "low"), cand(None, source="model-b"))
    assert merged.value is None and merged.unreadable and merged.partial == "INV-1O42"


def test_an_ungrounded_first_value_is_not_kept_as_confident():
    merged = merge_escalated(cand("INV-1042", "high", grounded=False), cand(None))
    assert merged.unreadable and merged.partial == "INV-1042"


def test_unreadable_from_either_side_stays_unreadable_with_the_best_partial():
    assert merge_escalated(cand(None, unreadable=True, partial="INV-10"), cand(None)).unreadable
    merged = merge_escalated(cand("INV-1042", "medium"), cand(None, unreadable=True))
    assert merged.unreadable and merged.partial == "INV-1042"


def test_an_invalid_second_reading_does_not_replace_a_valid_first_one():
    merged = merge_escalated(cand("INV-1042", "medium"), cand("??", issues=["bad"]))
    assert merged.value == "INV-1042"


def test_a_less_trusted_second_reading_never_replaces_a_trusted_first_one():
    # valid is not the same as true: the stronger model's value is not on the page
    merged = merge_escalated(cand("INV-1001", "high"),
                             cand("INV-9999", "low", grounded=False, source="model-b"))
    assert (merged.value, merged.source, merged.escalated) == ("INV-1001", "model-a", True)
    result = resolve_field(SCHEMA.field("invoice_number"), [merged], unread_pages=[])
    assert (result.status, result.value, result.confidence) == ("found", "INV-1001", "high")
    assert result.alternatives == [{"value": "INV-9999", "page": 1, "confidence": "low"}]


def test_a_replaced_first_value_is_kept_as_an_alternative():
    merged = merge_escalated(cand("INV-1O42", "low"), cand("INV-1042", source="model-b"))
    result = resolve_field(SCHEMA.field("invoice_number"), [merged], unread_pages=[])
    assert (result.value, result.source) == ("INV-1042", "model-b")
    assert result.alternatives == [{"value": "INV-1O42", "page": 1, "confidence": "low"}]


def test_missing_sides():
    assert merge_escalated(None, None) is None
    assert merge_escalated(None, cand("INV-1042")).value == "INV-1042"
    assert merge_escalated(cand("INV-1042"), None).value == "INV-1042"


FIRSTS = [cand("A", "high"), cand("A", "medium"), cand("A", "low"), cand("A", grounded=False),
          cand(None, unreadable=True, partial="A"), cand(None), cand("A", issues=["x"]), None]
SECONDS = [cand("B", "high"), cand("B", "low"), cand(None), cand(None, unreadable=True),
           cand("B", issues=["x"]), None]


def trust(c):
    return (c.grounded is not False, ["none", "low", "medium", "high"].index(c.confidence))


@pytest.mark.parametrize("first, second", list(itertools.product(FIRSTS, SECONDS)))
def test_escalation_never_turns_something_seen_into_not_on_the_document(first, second):
    merged = merge_escalated(first, second)
    if first is not None and first.seen:
        assert merged is not None and merged.seen
    if first is not None and first.usable and merged is not None and merged.usable:
        # a value is never traded for a less trusted one (ungrounded, or lower confidence)
        assert trust(merged) >= trust(first)
    if first is not None and second is not None and first.usable and second.usable:
        # and neither reading disappears: the one not chosen is listed as an alternative
        kept = {merged.value} | {alt.value for alt in merged.alternatives}
        assert {first.value, second.value} <= kept


# -- resolving one field across pages ----------------------------------------------------------

def at(page, value, confidence="high", grounded=True, **kw):
    c = cand(value, confidence, grounded=grounded, **kw)
    c.page = page
    return c


def test_the_best_candidate_wins_and_others_are_listed_as_alternatives():
    result = resolve_field(SCHEMA.field("invoice_number"), [
        at(1, "INV-1042", "low", grounded=False), at(2, "INV-1042", "high"),
        at(3, "INV-2000", "medium")], unread_pages=[])
    assert (result.status, result.value, result.page) == ("found", "INV-1042", 2)
    assert result.alternatives == [{"value": "INV-2000", "page": 3, "confidence": "medium"}]


def test_ties_go_to_the_earliest_page():
    result = resolve_field(SCHEMA.field("invoice_number"),
                           [at(2, "INV-B"), at(1, "INV-A")], unread_pages=[])
    assert result.value == "INV-A"


def test_unreadable_is_reported_with_its_partial_reading():
    result = resolve_field(SCHEMA.field("invoice_number"),
                           [at(1, None, unreadable=True, partial="INV-10")], unread_pages=[])
    assert (result.status, result.value, result.partial) == ("unreadable", None, "INV-10")


def test_invalid_values_are_reported_but_never_put_in_the_data():
    result = resolve_field(SCHEMA.field("invoice_date"),
                           [at(1, "next Tuesday", issues=["/invoice_date: not a valid date"])],
                           unread_pages=[])
    assert result.status == "invalid" and result.value is None
    assert result.raw == "next Tuesday" and result.issues == ["/invoice_date: not a valid date"]


def test_not_found_is_only_claimed_when_every_page_was_read():
    field = SCHEMA.field("po_number")
    assert resolve_field(field, [at(1, None)], unread_pages=[]).status == "not_found"
    unchecked = resolve_field(field, [at(1, None)], unread_pages=[2, 3])
    assert unchecked.status == "not_checked" and unchecked.pages_unread == [2, 3]


def test_a_found_value_does_not_care_about_unread_pages():
    assert resolve_field(SCHEMA.field("invoice_number"), [at(1, "INV-1042")],
                         unread_pages=[2]).status == "found"


def test_result_dict_omits_empty_details():
    result = resolve_field(SCHEMA.field("invoice_number"), [at(1, "INV-1042")], unread_pages=[])
    assert result.to_dict() == {"status": "found", "value": "INV-1042", "confidence": "high",
                                "page": 1, "evidence": None, "source": "model-a",
                                "grounded": True}
