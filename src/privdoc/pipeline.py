"""The extraction pipeline.

    plan     choose engines within the privacy tier (refusals happen here, before any I/O)
    load     a path, bytes, or an http(s) link through the SSRF-safe fetcher
    route    per page: text layer, OCR, or blank
    read     OCR / vision pages with the first rung of the reader ladder (in parallel, cached)
    extract  fields from page text (LLM or rules) and from vision readings
    repair   coerce, validate, and give an LLM one chance to fix what is still invalid
    escalate re-read only failed or doubtful pages with the next, stronger rung
    assemble schema-valid ``data`` plus per-field status, per-page routing and usage
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlsplit

from .cache import Cache, cache_key
from .config import Settings
from .document import open_document, parse_page_spec
from .engines.base import FieldExtractor, PageReader, PageText
from .engines.rules import RuleExtractor
from .errors import DocumentError, EngineError, PrivacyViolation, UsageError
from .escalation import PageView, choose_pages
from .fetch import FetchPolicy, SafeFetcher
from .fields import (
    CONFIDENCE_RANK,
    Candidate,
    FieldResult,
    merge_escalated,
    normalize_candidate,
    resolve_field,
)
from .plan import Plan, build_plan
from .privacy import Tier, parse_tier
from .prompts import PROMPT_VERSION
from .records import FieldOutput, PageReading, Usage
from .routing import Route, RoutingRules, route_page
from .schema import DocSchema, load_schema
from .transport import Transport

MB = 1024 * 1024
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ExtractOptions:
    tier: Tier | str | None = None  # None: the configured default (local unless set)
    engine: str = "auto"
    extractor: str = "auto"
    pages: str | None = None
    force_ocr: bool = False
    escalate: bool = True
    use_cache: bool = True
    include_text: bool = False


@dataclass
class PageResult:
    page: int
    route: str | None
    reason: str
    status: str  # read | blank | failed | skipped | no_engine | not_selected | page_limit
    reader: str | None = None
    legibility: str | None = None
    error: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    escalation_error: str | None = None
    cached: bool = False
    seconds: float = 0.0
    text: str = ""

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"page": self.page, "route": self.route, "reason": self.reason,
                               "status": self.status, "reader": self.reader,
                               "legibility": self.legibility}
        for key in ("error", "escalation_reason", "escalation_error"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.escalated:
            out["escalated"] = True
        if self.cached:
            out["cached"] = True
        out["seconds"] = round(self.seconds, 3)
        if include_text:
            out["text"] = self.text
        return out


@dataclass
class ExtractionResult:
    tier: str
    source: dict[str, Any]
    data: dict[str, Any]
    errors: list[dict[str, str]]
    fields: dict[str, FieldResult]
    pages: list[PageResult]
    pages_unread: list[dict[str, Any]]
    engines: dict[str, list[str]]
    data_sent_to: list[str]
    usage: Usage
    escalations: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    include_text: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def complete(self) -> bool:
        return not self.pages_unread

    @property
    def ok(self) -> bool:
        """At least one page could be read."""
        return any(p.status in ("read", "blank") for p in self.pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "valid": self.valid, "complete": self.complete, "tier": self.tier,
            "data": self.data, "errors": self.errors,
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "pages": [p.to_dict(self.include_text) for p in self.pages],
            "pages_unread": self.pages_unread, "source": self.source, "engines": self.engines,
            "data_sent_to": self.data_sent_to, "usage": self.usage.to_dict(),
            "escalations": self.escalations, "cache_hits": self.cache_hits,
            "seconds": round(self.seconds, 3), "warnings": self.warnings,
        }


class Pipeline:
    def __init__(self, settings: Settings | None = None, *, transport: Transport | None = None,
                 tesseract_runner: Any = None, fetcher: Any = None, cache: Cache | None = None,
                 which: Callable[[str], str | None] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings or Settings()
        self.transport = transport
        self.tesseract_runner = tesseract_runner
        self._fetcher = fetcher
        self.cache = cache if cache is not None else Cache(
            self.settings.cache_dir, ttl_seconds=self.settings.cache_ttl_days * 86400)
        self.which = which
        self.clock = clock
        self.sleep = sleep

    @property
    def fetcher(self) -> Any:
        if self._fetcher is None:
            s = self.settings
            self._fetcher = SafeFetcher(FetchPolicy(max_bytes=int(s.fetch_max_mb * MB),
                                                    timeout=s.fetch_timeout_s))
        return self._fetcher

    def plan(self, tier: Tier | None = None, engine: str = "auto",
             extractor: str = "auto") -> Plan:
        return build_plan(self.settings, tier, engine=engine, extractor=extractor,
                          transport=self.transport, tesseract_runner=self.tesseract_runner,
                          which=self.which, sleep=self.sleep)

    def extract(self, source: str | os.PathLike[str] | bytes, schema: DocSchema | str,
                options: ExtractOptions | None = None) -> ExtractionResult:
        opts = options or ExtractOptions()
        tier = parse_tier(opts.tier) if isinstance(opts.tier, str) else (
            opts.tier or self.settings.tier)
        doc_schema = schema if isinstance(schema, DocSchema) else load_schema(schema)
        started = self.clock()
        plan = self.plan(tier, opts.engine, opts.extractor)  # privacy refusals come first
        name, data, downloaded_from = self._load(source, tier)
        max_bytes = int(self.settings.max_file_mb * MB)
        with open_document(data, name, max_bytes=max_bytes) as doc:
            result = _Run(self, plan, doc, doc_schema, opts, started).execute()
        result.source["downloaded_from"] = downloaded_from
        return result

    def _load(self, source: str | os.PathLike[str] | bytes,
              tier: Tier) -> tuple[str, bytes, str | None]:
        if isinstance(source, bytes | bytearray):
            return "document", bytes(source), None
        text = os.fspath(source)
        if text.lower().startswith(("http://", "https://")):
            if tier is Tier.LOCAL:
                raise PrivacyViolation("the local tier makes no network calls, so it cannot "
                                       "download links; download the file and pass its path")
            fetched = self.fetcher.fetch(text)
            name = Path(unquote(urlsplit(fetched.final_url).path)).name or "download"
            return name, fetched.data, fetched.final_host
        path = Path(text)
        if not path.is_file():
            raise UsageError(f"file not found: {path}")
        limit = int(self.settings.max_file_mb * MB)
        if path.stat().st_size > limit:
            raise DocumentError(f"the file is larger than the {self.settings.max_file_mb:g} MB "
                                "limit")
        return path.name, path.read_bytes(), None


@dataclass
class _Page:
    number: int
    route: Route | None
    reason: str
    status: str = "pending"
    text: str = ""
    reader: str | None = None
    reading: PageReading | None = None
    rung: int = 0
    extract_rung: int = 0
    error: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    escalation_error: str | None = None
    seconds: float = 0.0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Run:
    """State for one extraction."""

    def __init__(self, pipeline: Pipeline, plan: Plan, doc: Any, schema: DocSchema,
                 opts: ExtractOptions, started: float) -> None:
        self.p = pipeline
        self.s = pipeline.settings
        self.plan = plan
        self.doc = doc
        self.schema = schema
        self.opts = opts
        self.started = started
        self.deadline = started + self.s.timeout_s
        self.pages: dict[int, _Page] = {}
        self.unread: dict[int, str] = {}
        self.cands: dict[str, list[Candidate]] = {f.name: [] for f in schema.fields}
        self.usage = Usage()
        self.cache_hits = 0
        self.escalations = 0
        self.warnings = list(plan.notes)
        self.use_cache = opts.use_cache and pipeline.cache.enabled

    # -- driver --------------------------------------------------------------------------------

    def execute(self) -> ExtractionResult:
        total = self.doc.page_count
        selected = parse_page_spec(self.opts.pages, total)
        chosen = set(selected[:self.s.max_pages])
        capped = set(selected[self.s.max_pages:])
        rules = RoutingRules(self.s.min_text_chars, self.s.scan_cover, self.opts.force_ocr)
        for n in range(1, total + 1):
            if n not in chosen:
                limited = n in capped
                self.pages[n] = _Page(n, None, "", "page_limit" if limited else "not_selected")
                self.unread[n] = (f"not read: beyond the {self.s.max_pages}-page limit (use "
                                  "--pages to choose pages)" if limited
                                  else "not selected with --pages")
                continue
            facts = self.doc.facts(n)
            route, reason = route_page(facts, rules)
            page = self.pages[n] = _Page(n, route, reason)
            if route is Route.TEXT:
                page.status, page.text, page.reader = "read", facts.text, "text-layer"
            elif route is Route.BLANK:
                page.status = "blank"

        ocr = [n for n in sorted(chosen) if self.pages[n].route is Route.OCR]
        if ocr and not self.plan.readers:
            for n in ocr:
                self.pages[n].status = "no_engine"
                self.unread[n] = "scanned page, and no OCR engine is available in this tier"
        elif ocr:
            self._read(ocr)

        self._initial_candidates()
        self._repair()
        results = self._resolve()
        if self.opts.escalate and self.s.max_escalations > 0:
            targets = choose_pages(self._views(), results, self.schema, self.s.max_escalations)
            if targets:
                self._escalate(targets)
                results = self._resolve()
        return self._assemble(results)

    # -- reading pages -------------------------------------------------------------------------

    def _map(self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        workers = min(self.s.concurrency, len(items))
        if workers <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="privdoc") as pool:
            return list(pool.map(fn, items))

    def _read_page(self, job: tuple[int, PageReader]) -> tuple[PageReading | None, str | None]:
        """(reading, None) | (None, error) | (None, None) when out of time."""
        number, reader = job
        if self.p.clock() > self.deadline:
            return None, None
        key = cache_key(kind="page", prompt=PROMPT_VERSION, reader=reader.fingerprint,
                        schema=self.schema.fingerprint if reader.extracts_fields else None,
                        doc=self.doc.sha256, page=number)
        if self.use_cache:
            hit = self.p.cache.get(key)
            if hit is not None:
                try:
                    return PageReading.from_cache(hit), None
                except (KeyError, TypeError):
                    pass
        try:
            image = self.doc.render(number, reader.render_dpi)
            reading = reader.read(image, number, self.doc.page_count, self.schema)
        except (EngineError, DocumentError) as err:
            return None, str(err)
        if self.use_cache:
            self.p.cache.put(key, reading.to_cache())
        return reading, None

    def _accept(self, page: _Page, reading: PageReading) -> None:
        page.reading, page.text, page.reader = reading, reading.text, reading.reader
        page.status, page.error = "read", None
        self.unread.pop(page.number, None)
        if reading.cached:
            self.cache_hits += 1
        else:
            page.seconds += reading.seconds
            self.usage = self.usage + reading.usage

    def _read(self, numbers: list[int]) -> None:
        reader = self.plan.readers[0]
        outcomes = self._map(self._read_page, [(n, reader) for n in numbers])
        for n, (reading, error) in zip(numbers, outcomes, strict=True):
            page = self.pages[n]
            if reading is not None:
                self._accept(page, reading)
            elif error is None:
                page.status = "skipped"
                self.unread[n] = "skipped: the time limit was reached"
            else:
                page.status, page.error = "failed", error
                self.unread[n] = f"could not be read: {error}"

    # -- fields --------------------------------------------------------------------------------

    def _extract(self, extractor: FieldExtractor, pages: list[PageText], *,
                 fallback: bool) -> tuple[FieldOutput, str]:
        cacheable = self.use_cache and extractor.destination is not None
        key = cache_key(kind="fields", prompt=PROMPT_VERSION, extractor=extractor.fingerprint,
                        schema=self.schema.fingerprint,
                        pages=[[p.page, _sha(p.text)] for p in pages])
        if cacheable:
            hit = self.p.cache.get(key)
            if hit is not None and isinstance(hit.get("fields"), dict):
                self.cache_hits += 1
                return FieldOutput(fields=hit["fields"], cached=True), extractor.id
        try:
            out = extractor.extract(pages, self.schema)
        except EngineError as err:
            if not fallback:
                raise
            self.warnings.append(f"field extraction with {extractor.id} failed ({err}); fell "
                                 "back to the rule extractor")
            return RuleExtractor(self.s.date_order).extract(pages, self.schema), "rules"
        self.usage = self.usage + out.usage
        if out.truncated:
            self.warnings.append("the document text was too long for one request and was "
                                 "truncated")
        if cacheable:
            self.p.cache.put(key, {"fields": out.fields})
        return out, extractor.id

    def _candidates(self, raw_fields: dict[str, Any], pages: list[PageText],
                    source: str) -> dict[str, Candidate]:
        texts = {p.page: p.text for p in pages}
        joined = "\n".join(p.text for p in pages)
        out = {}
        for spec in self.schema.fields:
            raw = raw_fields.get(spec.name)
            page = raw.get("page") if isinstance(raw, dict) else None
            if isinstance(page, str) and page.strip().isdigit():
                page = int(page)
            if isinstance(page, bool) or not isinstance(page, int) or page not in texts:
                page = pages[0].page if len(pages) == 1 else None
            out[spec.name] = normalize_candidate(
                raw, spec, self.schema, page=page, page_text=texts.get(page, joined),
                source=source, date_order=self.s.date_order)
        return out

    def _add(self, candidates: dict[str, Candidate]) -> None:
        for name, cand in candidates.items():
            if cand.seen:
                self.cands[name].append(cand)

    def _initial_candidates(self) -> None:
        text_pages = [PageText(n, p.text) for n, p in sorted(self.pages.items())
                      if p.status == "read" and (p.route is Route.TEXT or (
                          p.reading is not None and p.reading.fields is None))]
        if any(p.text.strip() for p in text_pages):
            out, source = self._extract(self.plan.extractors[0], text_pages, fallback=True)
            self._add(self._candidates(out.fields, text_pages, source))
        for n, page in sorted(self.pages.items()):
            if page.status == "read" and page.reading is not None and \
                    page.reading.fields is not None:
                self._add(self._candidates(page.reading.fields, [PageText(n, page.text)],
                                           page.reading.reader))

    def _repair(self) -> None:
        extractor = self.plan.extractors[0]
        broken = [c for cands in self.cands.values() for c in cands
                  if c.value is not None and c.issues]
        if not broken or not extractor.can_repair:
            return
        numbers = sorted({c.page for c in broken if c.page is not None})
        pages = [PageText(n, self.pages[n].text) for n in numbers] or [
            PageText(n, p.text) for n, p in sorted(self.pages.items())
            if p.status == "read" and p.text]
        previous = {c.field: {"value": c.raw, "page": c.page, "evidence": c.evidence}
                    for c in broken}
        problems = [issue for c in broken for issue in c.issues]
        try:
            out = extractor.repair(pages, self.schema, previous, problems)
        except EngineError as err:
            self.warnings.append(f"the repair round failed ({err})")
            return
        self.usage = self.usage + out.usage
        fixed = self._candidates(out.fields, pages, f"{extractor.id} (repaired)")
        for cand in broken:
            new = fixed.get(cand.field)
            if new is not None and new.usable:
                self.cands[cand.field] = [new if c is cand else c for c in self.cands[cand.field]]

    def _resolve(self) -> dict[str, FieldResult]:
        unread = sorted(self.unread)
        return {spec.name: resolve_field(spec, self.cands[spec.name], unread)
                for spec in self.schema.fields}

    # -- escalation ----------------------------------------------------------------------------

    def _views(self) -> list[PageView]:
        views = []
        for n, page in sorted(self.pages.items()):
            if page.route is Route.OCR:
                has_next = page.status in ("read", "failed") and \
                    page.rung + 1 < len(self.plan.readers)
            elif page.route is Route.TEXT:
                has_next = page.extract_rung + 1 < len(self.plan.extractors)
            else:
                has_next = False
            legibility = page.reading.legibility if page.reading is not None else None
            views.append(PageView(n, page.route, page.status, legibility, has_next))
        return views

    def _merge(self, number: int, new: dict[str, Candidate]) -> None:
        for spec in self.schema.fields:
            current = self.cands[spec.name]
            mine = [c for c in current if c.page == number]
            first = max(mine, key=lambda c: (c.usable, CONFIDENCE_RANK[c.confidence]),
                        default=None)
            merged = merge_escalated(first, new.get(spec.name))
            kept = [c for c in current if c.page != number]
            self.cands[spec.name] = kept + ([merged] if merged is not None and merged.seen
                                            else [])

    def _escalate(self, targets: list[tuple[int, str]]) -> None:
        for number, reason in targets:
            page = self.pages[number]
            page.escalated, page.escalation_reason = True, reason
            self.escalations += 1
        ocr = [n for n, _ in targets if self.pages[n].route is Route.OCR]
        jobs = [(n, self.plan.readers[self.pages[n].rung + 1]) for n in ocr]
        for (number, _), (reading, error) in zip(jobs, self._map(self._read_page, jobs),
                                                 strict=True):
            page = self.pages[number]
            page.rung += 1
            if reading is None:
                page.escalation_error = error or "skipped: the time limit was reached"
                continue
            self._accept(page, reading)
            single = [PageText(number, reading.text)]
            if reading.fields is not None:
                new = self._candidates(reading.fields, single, reading.reader)
            else:
                out, source = self._extract(self.plan.extractors[0], single, fallback=True)
                new = self._candidates(out.fields, single, source)
            self._merge(number, new)

        for number in [n for n, _ in targets if self.pages[n].route is Route.TEXT]:
            page = self.pages[number]
            page.extract_rung += 1
            if self.p.clock() > self.deadline:
                page.escalation_error = "skipped: the time limit was reached"
                continue
            extractor = self.plan.extractors[page.extract_rung]
            single = [PageText(number, page.text)]
            try:
                out, source = self._extract(extractor, single, fallback=False)
            except EngineError as err:
                page.escalation_error = str(err)
                continue
            self._merge(number, self._candidates(out.fields, single, source))

    # -- output --------------------------------------------------------------------------------

    def _assemble(self, results: dict[str, FieldResult]) -> ExtractionResult:
        data = {name: r.value for name, r in results.items() if r.status == "found"}
        errors = []
        for issue in self.schema.validate(data):
            message = issue.message
            name = issue.path.strip("/").split("/")[0]
            if message == "required field is missing" and name in results:
                message = f"required field is missing (status: {results[name].status})"
            errors.append({"path": issue.path, "message": message})
        pages = []
        for n, page in sorted(self.pages.items()):
            reading = page.reading
            pages.append(PageResult(
                page=n, route=str(page.route) if page.route else None, reason=page.reason,
                status=page.status, reader=page.reader,
                legibility=reading.legibility if reading is not None else (
                    "good" if page.route is Route.TEXT else None),
                error=page.error, escalated=page.escalated,
                escalation_reason=page.escalation_reason,
                escalation_error=page.escalation_error,
                cached=bool(reading is not None and reading.cached),
                seconds=page.seconds, text=page.text))
        return ExtractionResult(
            tier=str(self.plan.tier),
            source={"name": self.doc.name, "kind": self.doc.kind, "sha256": self.doc.sha256,
                    "pages_total": self.doc.page_count, "downloaded_from": None},
            data=data, errors=errors, fields=results, pages=pages,
            pages_unread=[{"page": n, "reason": r} for n, r in sorted(self.unread.items())],
            engines={"ocr": [r.id for r in self.plan.readers],
                     "extract": [e.id for e in self.plan.extractors]},
            data_sent_to=self.plan.gate.data_sent_to, usage=self.usage,
            escalations=self.escalations, cache_hits=self.cache_hits,
            seconds=self.p.clock() - self.started, warnings=self.warnings,
            include_text=self.opts.include_text)
