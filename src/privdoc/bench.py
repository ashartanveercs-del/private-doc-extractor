"""Benchmark harness: field accuracy, hallucination rate, cost and seconds per page.

Definitions (per field the schema asks for):

  present         the ground truth has a value
  absent          the field is not on the document (decoy fields included)
  correct         present, and the returned value equals the truth (by type: text ignoring case
                  and punctuation, numbers to the cent, dates by calendar day)
  wrong           present, but a different value came back
  missed          present, but nothing came back
  invented        absent, but a value came back
  unsupported     a returned value that does not occur anywhere in the document's real text:
                  every invented value, and wrong values that are not a misplaced real value.
                  "Occurs" is decided by :func:`occurs_in`, not by the pipeline's own grounding
                  check, and is stricter: whole printed words and numbers only

  field accuracy      = correct / present
  hallucination rate  = unsupported / returned values
  cost per page       = total estimated cost / pages      (n/a when a price is not configured)
  seconds per page    = total wall time / pages

Synthetic documents come from :mod:`privdoc.synth`, so the ground truth is exact and no real
document is involved. FUNSD can be added with ``--dataset funsd`` (see scripts/).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .coerce import find_dates, parse_number
from .errors import PrivacyViolation, PrivdocError, UnsafeURL, UsageError
from .grounding import normalize_text
from .schema import DocSchema, load_schema, primary_type
from .synth import KINDS, VARIANTS, make_doc, render

# punctuation that may sit around a printed word ("Total:", "(paid)", "Co.,") but not inside it
_AROUND_WORDS = "\"'`()[]{}<>,.;:!?*#|/\\-–—"
_AROUND_NUMBERS = "\"'`[]{}<>,;:!?*#|/\\"  # keeps "(12.50)", "-3" and "$5.00" readable


def _printed_words(text: str) -> list[str]:
    words = (word.strip(_AROUND_WORDS) for word in text.casefold().split())
    return [word for word in words if word]


def occurs_in(value: Any, text: str, node: Mapping[str, Any]) -> bool:
    """Is a returned value written in the document's real text?

    Used to score hallucinations, so it is deliberately independent of grounding.py (the
    pipeline's own check, which decides confidence) and stricter than it. Text must appear as
    the same sequence of whole whitespace-separated words, ignoring case and the punctuation
    around words but not inside them ("INV1042" is not "INV-1042", "INV" is not "INV-1042");
    a number must be a whole printed number, to the cent; a date the same calendar day. A
    boolean cannot be traced to text, so a wrong one is never "on the page".
    """
    if value is None:
        return True
    kind = primary_type(node)
    if kind == "array":
        return isinstance(value, list) and all(occurs_in(v, text, node["items"]) for v in value)
    if kind == "object":
        props = node["properties"]
        return isinstance(value, Mapping) and all(
            occurs_in(v, text, props[k]) for k, v in value.items() if k in props)
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        for word in text.split():
            number = parse_number(word.strip(_AROUND_NUMBERS))
            if number is None:
                number = parse_number(word.strip(_AROUND_WORDS + "$€£"))
            if number is not None and abs(number - value) < 0.005:
                return True
        return False
    if not isinstance(value, str):
        return False
    if node.get("format") == "date" and value in find_dates(text):
        return True
    wanted, words = _printed_words(value), _printed_words(text)
    return bool(wanted) and any(words[i:i + len(wanted)] == wanted
                                for i in range(len(words) - len(wanted) + 1))


@dataclass
class BenchCase:
    id: str
    kind: str
    variant: str
    data: bytes
    schema: DocSchema
    truth: dict[str, Any]
    source_text: str
    pages: int

    @property
    def asked(self) -> list[str]:
        return [f.name for f in self.schema.fields]


@dataclass
class Outcome:
    data: dict[str, Any] | None
    cost_usd: float | None = 0.0
    seconds: float = 0.0
    pages: int = 1
    error: str | None = None

    @classmethod
    def from_result(cls, result: Any) -> Outcome:
        return cls(data=result.data, cost_usd=result.usage.cost_usd, seconds=result.seconds,
                   pages=result.source["pages_total"])


@dataclass
class CaseScore:
    case_id: str
    kind: str
    variant: str
    present: int = 0
    absent: int = 0
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    invented: int = 0
    returned: int = 0
    unsupported: int = 0
    pages: int = 1
    cost_usd: float | None = 0.0
    seconds: float = 0.0
    error: str | None = None
    wrong_fields: list[str] = field(default_factory=list)
    missed_fields: list[str] = field(default_factory=list)
    invented_fields: list[str] = field(default_factory=list)


def values_equal(a: Any, b: Any, node: Mapping[str, Any]) -> bool:
    kind = primary_type(node)
    if a is None or b is None:
        return a is None and b is None
    if kind == "array":
        return (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                and all(values_equal(x, y, node["items"]) for x, y in zip(a, b, strict=True)))
    if kind == "object":
        props = node["properties"]
        return (isinstance(a, Mapping) and isinstance(b, Mapping)
                and all(values_equal(a.get(k), b.get(k), props[k]) for k in props))
    if kind in ("number", "integer"):
        try:
            return abs(float(a) - float(b)) < 0.005
        except (TypeError, ValueError):
            return False
    if kind == "boolean" or node.get("format") == "date":
        return a == b
    return normalize_text(str(a)) == normalize_text(str(b))


def score_case(case: BenchCase, outcome: Outcome) -> CaseScore:
    score = CaseScore(case.id, case.kind, case.variant, pages=outcome.pages,
                      cost_usd=outcome.cost_usd, seconds=outcome.seconds, error=outcome.error)
    data = outcome.data or {}
    for spec in case.schema.fields:
        truth, got = case.truth.get(spec.name), data.get(spec.name)
        if truth is not None:
            score.present += 1
        else:
            score.absent += 1
        if got is None:
            if truth is not None:
                score.missed += 1
                score.missed_fields.append(spec.name)
            continue
        score.returned += 1
        if truth is None:
            score.invented += 1
            score.invented_fields.append(spec.name)
            score.unsupported += 1
        elif values_equal(got, truth, spec.schema):
            score.correct += 1
        else:
            score.wrong += 1
            score.wrong_fields.append(spec.name)
            if not occurs_in(got, case.source_text, spec.schema):
                score.unsupported += 1
    return score


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class BenchReport:
    rows: list[CaseScore]
    config: dict[str, Any] = field(default_factory=dict)

    def totals(self, rows: Iterable[CaseScore] | None = None) -> dict[str, Any]:
        rows = list(self.rows if rows is None else rows)
        pages = sum(r.pages for r in rows)
        costs = [r.cost_usd for r in rows]
        cost = None if any(c is None for c in costs) else sum(c or 0.0 for c in costs)
        present = sum(r.present for r in rows)
        returned = sum(r.returned for r in rows)
        return {
            "docs": len(rows), "pages": pages, "present": present,
            "absent": sum(r.absent for r in rows), "correct": sum(r.correct for r in rows),
            "wrong": sum(r.wrong for r in rows), "missed": sum(r.missed for r in rows),
            "invented": sum(r.invented for r in rows), "returned": returned,
            "unsupported": sum(r.unsupported for r in rows),
            "field_accuracy": _ratio(sum(r.correct for r in rows), present),
            "hallucination_rate": _ratio(sum(r.unsupported for r in rows), returned),
            "cost_usd": cost, "cost_per_page": None if cost is None else _ratio(cost, pages),
            "seconds_per_page": _ratio(sum(r.seconds for r in rows), pages),
            "errors": sum(1 for r in rows if r.error),
        }

    def by(self, key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[CaseScore]] = {}
        for row in self.rows:
            groups.setdefault(getattr(row, key), []).append(row)
        return {name: self.totals(rows) for name, rows in groups.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config, "totals": self.totals(),
                "by_variant": self.by("variant"), "by_kind": self.by("kind"),
                "rows": [asdict(r) for r in self.rows]}

    def to_markdown(self) -> str:
        head = ("| Slice | Docs | Pages | Field accuracy | Hallucination rate | "
                "Invented (absent fields) | Cost / page | Seconds / page | Errors |")
        lines = [head, "|---|---|---|---|---|---|---|---|---|"]
        slices = [("all", self.totals()), *self.by("variant").items(), *self.by("kind").items()]
        for name, t in slices:
            cost = "n/a" if t["cost_per_page"] is None else f"${t['cost_per_page']:.5f}"
            lines.append(
                f"| {name} | {t['docs']} | {t['pages']} | {t['field_accuracy']:.1%} "
                f"({t['correct']}/{t['present']}) | {t['hallucination_rate']:.1%} "
                f"({t['unsupported']}/{t['returned']}) | {t['invented']}/{t['absent']} | "
                f"{cost} | {t['seconds_per_page']:.2f} | {t['errors']} |")
        return "\n".join(lines)


def run_bench(cases: list[BenchCase], extract: Callable[[BenchCase], Outcome], *,
              config: dict[str, Any] | None = None,
              progress: Callable[[int, int, CaseScore], None] | None = None) -> BenchReport:
    rows = []
    for index, case in enumerate(cases, start=1):
        try:
            outcome = extract(case)
        except (PrivacyViolation, UsageError, UnsafeURL):
            raise  # a configuration problem, not a property of this document
        except PrivdocError as err:
            outcome = Outcome(data=None, error=str(err), pages=case.pages)
        score = score_case(case, outcome)
        rows.append(score)
        if progress:
            progress(index, len(cases), score)
    return BenchReport(rows, dict(config or {}))


def synthetic_cases(kinds: Iterable[str], variants: Iterable[str], per_kind: int,
                    seed: int) -> list[BenchCase]:
    cases = []
    for kind in kinds:
        schema = load_schema(kind)
        for doc_seed in range(seed, seed + per_kind):
            doc = make_doc(kind, doc_seed)
            for variant in variants:
                rendered = render(doc, variant)
                cases.append(BenchCase(
                    id=f"{kind}-{doc_seed}-{variant}", kind=kind, variant=variant,
                    data=rendered.data, schema=schema, truth=doc.truth, source_text=doc.text,
                    pages=rendered.pages))
    return cases


def _choices(raw: str, allowed: Iterable[str], what: str) -> list[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [v for v in values if v not in allowed]
    if unknown or not values:
        raise UsageError(f"unknown {what} {unknown or raw!r}; choose from " + ", ".join(allowed))
    return values


def run_cli(args: Any, settings: Any, kwargs: dict[str, Any]) -> int:
    from .datasets import find_funsd_root, load_funsd
    from .pipeline import ExtractOptions, Pipeline
    from .privacy import parse_tier

    tier = parse_tier(args.tier) if args.tier else settings.tier
    if args.dataset == "funsd":
        if not args.dataset_dir:
            raise UsageError("--dataset funsd needs --dataset-dir (see scripts/download_funsd.py)")
        cases = load_funsd(find_funsd_root(Path(args.dataset_dir)), limit=args.limit)
    else:
        cases = synthetic_cases(_choices(args.kinds, KINDS, "kind"),
                                _choices(args.variants, VARIANTS, "variant"),
                                args.per_kind, args.seed)
    pipeline = Pipeline(settings, **kwargs)
    options = ExtractOptions(tier=tier, engine=args.engine, extractor=args.extractor,
                             escalate=not args.no_escalate, use_cache=False)
    plan = pipeline.plan(tier, args.engine, args.extractor).describe()
    config = {"tier": str(tier), "dataset": args.dataset, "engine": args.engine,
              "extractor": args.extractor, "escalate": not args.no_escalate,
              "ocr": plan["ocr"], "extract": plan["extract"], "cases": len(cases)}

    def extract(case: BenchCase) -> Outcome:
        return Outcome.from_result(pipeline.extract(case.data, case.schema, options))

    def progress(index: int, total: int, score: CaseScore) -> None:
        status = f"error: {score.error}" if score.error else f"{score.correct}/{score.present}"
        print(f"[{index}/{total}] {score.case_id}: {status}", file=sys.stderr)

    report = run_bench(cases, extract, config=config, progress=progress)
    print(report.to_markdown())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2) + "\n",
                                       encoding="utf-8")
    return 0
