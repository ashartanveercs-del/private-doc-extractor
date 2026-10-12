import json

import pytest

from privdoc.bench import (
    BenchCase,
    Outcome,
    run_bench,
    score_case,
    synthetic_cases,
    values_equal,
)
from privdoc.cli import main
from privdoc.config import Settings
from privdoc.errors import DocumentError, PrivacyViolation
from privdoc.pipeline import ExtractOptions, Pipeline
from privdoc.privacy import Tier
from privdoc.schema import DocSchema

SCHEMA = DocSchema({"type": "object", "properties": {
    "number": {"type": "string"}, "total": {"type": "number"},
    "date": {"type": "string", "format": "date"}, "po": {"type": "string"},
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "description": {"type": "string"}, "amount": {"type": "number"}}}}}})
SOURCE = "Invoice #: INV-1\nDate: 01/02/2026\nSubtotal: 9.00\nTotal: $10.00\nDesk lamp 9.00"


def case(truth, **kw):
    return BenchCase(id="c1", kind="invoice", variant="text", data=b"", schema=SCHEMA,
                     truth=truth, source_text=SOURCE, pages=1, **kw)


def test_scoring_separates_misses_wrong_values_and_hallucinations():
    truth = {"number": "INV-1", "total": 10.0, "date": "2026-01-02"}
    outcome = Outcome(data={"number": "inv 1", "total": 9.0, "po": "PO-77"}, cost_usd=0.002,
                      seconds=1.5, pages=1)
    score = score_case(case(truth), outcome)
    assert (score.present, score.absent) == (3, 2)
    assert (score.correct, score.wrong, score.missed, score.invented) == (1, 1, 1, 1)
    # 9.00 is on the page (the subtotal): a wrong pick, not a hallucination; PO-77 is invented
    assert (score.returned, score.unsupported) == (3, 1)
    assert score.wrong_fields == ["total"] and score.missed_fields == ["date"]
    assert score.invented_fields == ["po"]


def test_an_extraction_error_counts_every_present_field_as_missed():
    score = score_case(case({"number": "INV-1", "total": 10.0}),
                       Outcome(data=None, error="could not open the PDF", pages=1))
    assert (score.missed, score.returned, score.error) == (2, 0, "could not open the PDF")


@pytest.mark.parametrize("node, a, b, equal", [
    ({"type": "string"}, "Cedar Lane  Plumbing", "cedar lane plumbing", True),
    ({"type": "number"}, 10.0, 10.004, True),
    ({"type": "number"}, 10.0, 10.01, False),
    ({"type": "string", "format": "date"}, "2026-01-02", "2026-01-02", True),
    ({"type": "boolean"}, True, False, False),
    (SCHEMA.field("items").schema, [{"description": "Desk lamp", "amount": 9}],
     [{"description": "desk lamp", "amount": 9.0}], True),
    (SCHEMA.field("items").schema, [{"description": "Desk lamp", "amount": 9}], [], False),
])
def test_values_are_compared_by_type(node, a, b, equal):
    assert values_equal(a, b, node) is equal


def test_the_report_aggregates_and_slices():
    cases = [case({"number": "INV-1", "total": 10.0}),
             BenchCase(id="c2", kind="receipt", variant="scan", data=b"", schema=SCHEMA,
                       truth={"number": "INV-1"}, source_text=SOURCE, pages=2)]
    outcomes = {"c1": Outcome(data={"number": "INV-1", "total": 10.0}, cost_usd=0.001,
                              seconds=1.0, pages=1),
                "c2": Outcome(data={"number": "INV-2"}, cost_usd=0.002, seconds=3.0, pages=2)}
    report = run_bench(cases, lambda c: outcomes[c.id], config={"tier": "cloud"})
    total = report.totals()
    assert total["field_accuracy"] == pytest.approx(2 / 3)
    assert total["hallucination_rate"] == pytest.approx(1 / 3)
    assert total["cost_per_page"] == pytest.approx(0.001)
    assert total["seconds_per_page"] == pytest.approx(4 / 3)
    assert set(report.by("variant")) == {"text", "scan"}
    markdown = report.to_markdown()
    assert "| all | 2 | 3 | 66.7% (2/3) | 33.3% (1/3) | 0/7 |" in markdown
    assert json.loads(json.dumps(report.to_dict()))["totals"]["docs"] == 2


def test_unknown_costs_are_reported_as_unknown():
    report = run_bench([case({"number": "INV-1"})],
                       lambda c: Outcome(data={"number": "INV-1"}, cost_usd=None, pages=1))
    assert report.totals()["cost_per_page"] is None
    assert "n/a" in report.to_markdown()


def test_errors_are_recorded_but_privacy_and_usage_problems_stop_the_run():
    def boom(c):
        raise DocumentError("damaged file")

    report = run_bench([case({"number": "INV-1"})], boom)
    assert report.rows[0].error == "damaged file" and report.totals()["errors"] == 1

    def refused(c):
        raise PrivacyViolation("no")

    with pytest.raises(PrivacyViolation):
        run_bench([case({"number": "INV-1"})], refused)


def test_the_synthetic_corpus_is_deterministic():
    a = synthetic_cases(["invoice", "receipt"], ["text", "scan"], per_kind=2, seed=5)
    b = synthetic_cases(["invoice", "receipt"], ["text", "scan"], per_kind=2, seed=5)
    assert len(a) == 8 and len({c.id for c in a}) == 8
    assert [c.truth for c in a] == [c.truth for c in b] and a[0].data == b[0].data
    assert a[0].asked == [f.name for f in a[0].schema.fields]


def test_typed_documents_score_perfectly_in_the_local_tier():
    cases = synthetic_cases(["invoice", "receipt", "tax_form"], ["text"], per_kind=2, seed=7)
    pipeline = Pipeline(Settings.from_env({"PRIVDOC_CACHE": "off"}), which=lambda c: None)

    def extract(c):
        result = pipeline.extract(c.data, c.schema, ExtractOptions(tier=Tier.LOCAL,
                                                                   use_cache=False))
        return Outcome.from_result(result)

    totals = run_bench(cases, extract).totals()
    assert totals["field_accuracy"] == 1.0 and totals["hallucination_rate"] == 0.0
    assert totals["cost_per_page"] == 0.0


def test_bench_command_prints_a_table_and_writes_json(capsys, tmp_path):
    out = tmp_path / "bench.json"
    code = main(["bench", "--tier", "local", "--kinds", "receipt", "--variants", "text,scan",
                 "--per-kind", "1", "--json", str(out)],
                env={"PRIVDOC_CACHE": "off"}, pipeline_kwargs={"which": lambda c: None})
    printed = capsys.readouterr().out
    assert code == 0 and "| Slice |" in printed and "| scan |" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["config"]["tier"] == "local" and len(report["rows"]) == 2
