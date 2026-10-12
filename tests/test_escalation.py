from privdoc.escalation import PageView, choose_pages
from privdoc.fields import FieldResult
from privdoc.routing import Route
from privdoc.schema import load_schema

SCHEMA = load_schema("receipt")


def results(**overrides):
    out = {f.name: FieldResult(f.name, "found", value="x", confidence="high", page=1,
                               grounded=True) for f in SCHEMA.fields}
    out.update({name: FieldResult(name, **kw) for name, kw in overrides.items()})
    return out


def view(n, route=Route.OCR, status="read", legibility="good", has_next=True):
    return PageView(n, route, status, legibility, has_next)


def test_nothing_to_do_when_every_field_is_confident():
    assert choose_pages([view(1), view(2)], results(), SCHEMA, budget=3) == []


def test_priorities_failed_then_poor_then_field_problems_then_missing_required():
    pages = [view(1, legibility="fair"), view(2, legibility="poor"), view(3, status="failed"),
             view(4)]
    chosen = choose_pages(pages, results(
        tip={"status": "found", "value": 1.0, "confidence": "low", "page": 4, "grounded": True},
        total={"status": "not_checked"}), SCHEMA, budget=10)
    assert chosen == [(3, "the first reading failed"), (2, "poor legibility"),
                      (4, "tip: low confidence"), (1, "required field total is missing")]


def test_the_budget_keeps_the_most_urgent_pages():
    pages = [view(1, legibility="poor"), view(2, status="failed")]
    assert [n for n, _ in choose_pages(pages, results(), SCHEMA, budget=1)] == [2]
    assert choose_pages(pages, results(), SCHEMA, budget=0) == []


def test_pages_without_a_stronger_rung_are_never_chosen():
    pages = [view(1, status="failed", has_next=False), view(2, route=Route.TEXT, has_next=False)]
    chosen = choose_pages(pages, results(
        date={"status": "invalid", "page": 2, "raw": "soon"}), SCHEMA, budget=5)
    assert chosen == []


def test_text_pages_escalate_for_their_field_problems_but_not_for_missing_fields():
    pages = [view(1, route=Route.TEXT), view(2, route=Route.TEXT)]
    chosen = choose_pages(pages, results(
        merchant_name={"status": "found", "value": "Cafe", "confidence": "low", "page": 2,
                       "grounded": False},
        total={"status": "not_found"}), SCHEMA, budget=5)
    assert chosen == [(2, "merchant_name: not found in the page text")]
