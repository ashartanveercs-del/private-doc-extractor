import base64
import io

import pytest
from PIL import Image

from privdoc.config import Settings
from privdoc.errors import PrivacyViolation, UsageError
from privdoc.fetch import Fetched
from privdoc.pipeline import ExtractOptions, Pipeline
from privdoc.privacy import Tier
from privdoc.schema import DocSchema, load_schema
from privdoc.synth import PdfPage, TextRun, make_doc, render, write_pdf
from tests.fakes import FakeLLM, FakeTesseract, FakeTransport, fval
from tests.oracle import oracle_fields, oracle_page, page_texts

LLM_ENV = {"PRIVDOC_LLM_BASE_URL": "https://llm.example.com/v1", "PRIVDOC_LLM_API_KEY": "sk-t",
           "PRIVDOC_LLM_MODEL": "small", "PRIVDOC_LLM_ESCALATION_MODEL": "large",
           "PRIVDOC_PRICES": "small=0.1/0.4,large=2/8"}
CLOUD = ExtractOptions(tier=Tier.CLOUD)
LOCAL = ExtractOptions(tier=Tier.LOCAL)


def pipeline(env=None, *, llm=None, tesseract=None, tesseract_found=True, cache_dir=None,
             clock=None, fetcher=None):
    base = {"PRIVDOC_CONCURRENCY": "1", "PRIVDOC_CACHE": "off"}
    if cache_dir is not None:
        base.update({"PRIVDOC_CACHE": "on", "PRIVDOC_CACHE_DIR": str(cache_dir)})
    settings = Settings.from_env({**base, **(env or {})})
    transport = FakeTransport(llm or FakeLLM())
    extra = {"clock": clock} if clock else {}
    p = Pipeline(settings, transport=transport,
                 tesseract_runner=tesseract if tesseract is not None else FakeTesseract(),
                 which=lambda cmd: "/usr/bin/tesseract" if tesseract_found else None,
                 sleep=lambda s: None, fetcher=fetcher, **extra)
    return p, transport


def same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, int | float) and not isinstance(a, bool):
        return abs(a - b) < 0.005
    return a == b


def assert_matches_truth(result, doc):
    for name, value in doc.truth.items():
        assert same(result.data.get(name), value), (name, result.fields[name])
    assert load_schema(doc.kind).validate(result.data) == []


def invoice_with(field, start=0):
    return next(make_doc("invoice", s) for s in range(start, start + 100)
                if field in make_doc("invoice", s).truth)


# -- local tier --------------------------------------------------------------------------------

def test_typed_pdf_in_the_local_tier_is_read_from_its_text_layer_for_free():
    doc = make_doc("invoice", 3)
    tesseract = FakeTesseract()
    p, transport = pipeline(tesseract=tesseract)
    result = p.extract(render(doc, "text").data, "invoice", LOCAL)
    assert result.ok and result.valid and result.complete
    assert_matches_truth(result, doc)
    assert [page.route for page in result.pages] == ["text-layer", "text-layer"]
    assert result.usage.cost_usd == 0 and result.data_sent_to == []
    assert tesseract.calls == [] and transport.requests == []
    assert result.engines == {"ocr": ["tesseract:eng:psm3:300dpi",
                                      "tesseract:eng:psm6:400dpi+clean"], "extract": ["rules"]}


def test_the_local_tier_never_touches_the_network_even_with_keys_configured(no_network_at_all):
    doc = make_doc("receipt", 4)
    tesseract = FakeTesseract(pages={2: page_texts(doc, "mixed")[1]})
    p, transport = pipeline({**LLM_ENV, "PRIVDOC_GCV_API_KEY": "g"}, tesseract=tesseract)
    result = p.extract(render(doc, "mixed").data, "receipt", LOCAL)
    assert result.valid
    assert_matches_truth(result, doc)
    assert [page.reader for page in result.pages] == ["text-layer", "tesseract:eng:psm3:300dpi"]
    assert no_network_at_all.attempts == []
    assert transport.requests == [] and result.data_sent_to == []


def test_a_link_is_refused_in_the_local_tier_before_any_lookup(no_network_at_all):
    p, _ = pipeline()
    with pytest.raises(PrivacyViolation, match="local"):
        p.extract("https://docs.example.com/invoice.pdf", "invoice", LOCAL)
    assert no_network_at_all.attempts == []


def test_privacy_refusals_happen_before_the_file_is_even_opened():
    p, _ = pipeline(LLM_ENV)
    with pytest.raises(PrivacyViolation):
        p.extract("no-such-file.pdf", "invoice",
                  ExtractOptions(tier=Tier.LOCAL, engine="vision-llm"))
    with pytest.raises(UsageError, match="no-such-file.pdf"):
        p.extract("no-such-file.pdf", "invoice", LOCAL)


def test_without_an_ocr_engine_readable_pages_count_and_the_rest_are_not_checked():
    doc = make_doc("receipt", 4)
    p, _ = pipeline(tesseract_found=False)
    result = p.extract(render(doc, "mixed").data, "receipt", LOCAL)
    assert result.ok and not result.complete and not result.valid
    assert result.fields["merchant_name"].status == "found"
    assert result.fields["total"].status == "not_checked"
    assert result.fields["total"].pages_unread == [2]
    (unread,) = result.pages_unread
    assert unread["page"] == 2 and "no OCR engine" in unread["reason"]
    assert any("Tesseract" in w for w in result.warnings)
    assert any(e["path"] == "/total" and "not_checked" in e["message"] for e in result.errors)


def test_a_searchable_scan_is_ocred_instead_of_trusting_its_label_only_text_layer():
    doc = make_doc("tax_form", 2)
    tesseract = FakeTesseract(pages={1: page_texts(doc, "searchable-scan")[0]})
    p, _ = pipeline(tesseract=tesseract)
    result = p.extract(render(doc, "searchable-scan").data, "tax_form", LOCAL)
    assert result.pages[0].route == "ocr" and "covers" in result.pages[0].reason
    assert result.valid
    assert_matches_truth(result, doc)


def test_blank_pages_are_skipped_without_making_the_document_incomplete():
    data = write_pdf([PdfPage(runs=[TextRun(72, 100, "Store: Corner Bean Cafe"),
                                    TextRun(72, 120, "Date: 2026-03-14"),
                                    TextRun(72, 140, "Total: $5.00")]), PdfPage()])
    p, _ = pipeline()
    result = p.extract(data, "receipt", LOCAL)
    assert [page.status for page in result.pages] == ["read", "blank"]
    assert result.complete and result.valid
    assert result.fields["tip"].status == "not_found"


def test_page_selection_and_the_page_cap_make_unread_pages_not_checked():
    doc = make_doc("invoice", 3)
    data = render(doc, "text").data
    p, _ = pipeline()
    selected = p.extract(data, "invoice", ExtractOptions(tier=Tier.LOCAL, pages="1"))
    assert selected.fields["invoice_number"].status == "found"
    assert selected.fields["total"].status == "not_checked"
    assert "not selected" in selected.pages_unread[0]["reason"]
    capped, _ = pipeline({"PRIVDOC_MAX_PAGES": "1"})
    result = capped.extract(data, "invoice", LOCAL)
    assert "page limit" in result.pages_unread[0]["reason"]


def test_pages_left_when_time_runs_out_are_skipped_not_reported_missing():
    doc = make_doc("invoice", 3)
    now = [0.0]

    def slow_tesseract(cmd, timeout):
        now[0] += 400  # every OCR call blows the 300 s budget
        return FakeTesseract(pages={1: page_texts(doc, "scan")[0]})(cmd, timeout)

    p, _ = pipeline({"PRIVDOC_TIMEOUT": "300"}, tesseract=slow_tesseract,
                    clock=lambda: now[0])
    result = p.extract(render(doc, "scan").data, "invoice", LOCAL)
    assert [page.status for page in result.pages] == ["read", "skipped"]
    assert "time limit" in result.pages_unread[0]["reason"]
    assert result.fields["total"].status == "not_checked"


# -- cloud tier --------------------------------------------------------------------------------

def test_cloud_tier_uses_text_extraction_for_typed_pages_and_vision_for_scans():
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: oracle_page(doc, "mixed", n),
                  fields=lambda model, pages, text: oracle_fields(doc, "mixed", pages))
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "mixed").data, "invoice", CLOUD)
    assert result.valid
    assert_matches_truth(result, doc)
    assert [(c.kind, c.model) for c in llm.calls] == [("page_reading", "small"),
                                                      ("document_fields", "small")]
    assert llm.calls[0].page == 2 and llm.calls[1].pages == [1]
    assert [page.reader for page in result.pages] == ["text-layer", "vision-llm:small"]
    assert result.data_sent_to == ["LLM API (llm.example.com, third-party)"]
    assert result.usage.llm_calls == 2
    assert result.usage.cost_usd == pytest.approx(2 * (1000 * 0.1 + 200 * 0.4) / 1e6)


def test_only_doubtful_pages_are_re_read_by_the_stronger_model():
    doc = make_doc("invoice", 5)

    def page(model, n):
        if model == "small" and n == 2:
            return oracle_page(doc, "scan", 2, total=fval(doc.truth["total"], confidence="low"))
        return oracle_page(doc, "scan", n)

    llm = FakeLLM(page=page)
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "scan").data, "invoice", CLOUD)
    assert [(c.model, c.page) for c in llm.calls] == [("small", 1), ("small", 2), ("large", 2)]
    assert result.escalations == 1
    assert result.pages[1].escalated and not result.pages[0].escalated
    assert "total" in result.pages[1].escalation_reason
    total = result.fields["total"]
    assert (total.confidence, total.source, total.escalated) == ("high", "vision-llm:large", True)
    assert result.valid


def test_the_escalation_budget_caps_re_reads():
    doc = make_doc("invoice", 5)

    def page(model, n):
        if model == "small":
            name = "invoice_number" if n == 1 else "total"
            return oracle_page(doc, "scan", n, **{name: fval(doc.truth[name], confidence="low")})
        return oracle_page(doc, "scan", n)

    llm = FakeLLM(page=page)
    p, _ = pipeline({**LLM_ENV, "PRIVDOC_MAX_ESCALATIONS": "1"}, llm=llm)
    result = p.extract(render(doc, "scan").data, "invoice", CLOUD)
    assert [c.page for c in llm.calls if c.model == "large"] == [1]
    assert result.escalations == 1 and result.fields["total"].confidence == "low"


def test_no_escalation_when_asked_not_to():
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: oracle_page(
        doc, "scan", n, total=fval(doc.truth["total"], confidence="low")))
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "scan").data, "invoice",
                       ExtractOptions(tier=Tier.CLOUD, escalate=False))
    assert {c.model for c in llm.calls} == {"small"} and result.escalations == 0


def test_a_confident_value_survives_a_stronger_model_that_misses_it():
    doc = invoice_with("tax")

    def page(model, n):
        if n == 2 and model == "small":
            return oracle_page(doc, "scan", 2, tax=fval(doc.truth["tax"], confidence="low"))
        if n == 2 and model == "large":
            return oracle_page(doc, "scan", 2, total=fval(None))
        return oracle_page(doc, "scan", n)

    p, _ = pipeline(LLM_ENV, llm=FakeLLM(page=page))
    result = p.extract(render(doc, "scan").data, "invoice", CLOUD)
    assert result.fields["total"].status == "found"
    assert result.fields["total"].source == "vision-llm:small"
    assert result.fields["tax"].source == "vision-llm:large"
    assert result.valid


def test_a_failed_page_is_retried_with_the_stronger_model():
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: 500 if (model, n) == ("small", 2)
                  else oracle_page(doc, "scan", n))
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "scan").data, "invoice", CLOUD)
    assert result.complete and result.valid
    page2 = result.pages[1]
    assert page2.status == "read" and page2.escalated and page2.reader == "vision-llm:large"
    assert "failed" in page2.escalation_reason


def test_a_page_that_no_engine_can_read_is_unread_and_its_fields_not_checked():
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: 500 if n == 2 else oracle_page(doc, "scan", n))
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "scan").data, "invoice", CLOUD)
    assert result.ok and not result.complete and not result.valid
    assert result.pages[1].status == "failed" and "HTTP 500" in result.pages[1].error
    assert "could not be read" in result.pages_unread[0]["reason"]
    assert result.fields["total"].status == "not_checked"


def test_invalid_values_get_one_repair_round():
    doc = make_doc("invoice", 3)
    good = oracle_fields(doc, "text", [1, 2])["fields"]["invoice_date"]
    llm = FakeLLM(
        fields=lambda model, pages, text: oracle_fields(
            doc, "text", pages, invoice_date=fval("the fourteenth", page=1)),
        repair=lambda model, text: {"fields": {"invoice_date": {**good, "page": 1}}})
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "text").data, "invoice", CLOUD)
    assert [c.kind for c in llm.calls] == ["document_fields", "field_repair"]
    assert "not a valid date" in llm.calls[1].text
    assert result.data["invoice_date"] == doc.truth["invoice_date"] and result.valid


def test_values_still_invalid_after_repair_are_reported_but_never_used():
    doc = make_doc("invoice", 3)
    bad = fval("the fourteenth", page=1)
    llm = FakeLLM(fields=lambda model, pages, text: oracle_fields(doc, "text", pages,
                                                                  invoice_date=bad),
                  repair=lambda model, text: {"fields": {"invoice_date": bad}})
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "text").data, "invoice", CLOUD)
    field = result.fields["invoice_date"]
    assert field.status == "invalid" and field.raw == "the fourteenth" and field.issues
    assert "invoice_date" not in result.data and not result.valid


def test_a_value_that_is_not_on_the_page_is_re_checked_by_the_stronger_model():
    doc = make_doc("invoice", 3)

    def fields(model, pages, text):
        if model == "small":
            return oracle_fields(doc, "text", pages,
                                 invoice_number=fval("INV-0000000", evidence=None, page=1))
        return oracle_fields(doc, "text", pages)

    llm = FakeLLM(fields=fields)
    p, _ = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "text").data, "invoice", CLOUD)
    assert [(c.model, c.pages) for c in llm.calls] == [("small", [1, 2]), ("large", [1])]
    number = result.fields["invoice_number"]
    assert (number.value, number.source) == (doc.truth["invoice_number"], "text-llm:large")


ORDER = DocSchema({"type": "object", "required": ["order_number"], "properties": {
    "order_number": {"type": "string", "x-labels": ["Order #"]},
    "items": {"type": "array", "items": {
        "type": "object", "required": ["name", "quantity"], "properties": {
            "name": {"type": "string"}, "quantity": {"type": "integer"},
            "sku": {"type": "string"}}}}}})


def test_an_optional_nested_property_the_model_sends_as_null_is_absent_not_invalid():
    lines = ["Order #: PO-3301", "Desk lamp 2", "USB-C cable 3 SKU C-77"]
    data = write_pdf([PdfPage(runs=[TextRun(72, 100 + 20 * i, line)
                                    for i, line in enumerate(lines)])])
    items = [{"name": "Desk lamp", "quantity": 2, "sku": None},
             {"name": "USB-C cable", "quantity": 3, "sku": "C-77"}]
    llm = FakeLLM(fields=lambda model, pages, text: {"fields": {
        "order_number": fval("PO-3301", evidence="Order #: PO-3301", page=1),
        "items": fval(items, evidence="Desk lamp 2", page=1)}})
    p, transport = pipeline(LLM_ENV, llm=llm)
    result = p.extract(data, ORDER, CLOUD)
    request = transport.requests[0].body["response_format"]["json_schema"]["schema"]
    item = request["properties"]["fields"]["properties"]["items"]["properties"]["value"]["items"]
    assert item["properties"]["sku"]["type"] == ["string", "null"]  # strict form forces null
    assert result.fields["items"].status == "found" and result.valid
    assert result.data["items"] == [{"name": "Desk lamp", "quantity": 2},
                                    {"name": "USB-C cable", "quantity": 3, "sku": "C-77"}]
    assert [c.kind for c in llm.calls] == ["document_fields"]  # no repair, no escalation


def test_a_failed_text_extraction_falls_back_to_the_rule_extractor():
    doc = make_doc("receipt", 1)
    p, _ = pipeline(LLM_ENV, llm=FakeLLM(fields=lambda model, pages, text: 503))
    result = p.extract(render(doc, "text").data, "receipt", CLOUD)
    assert result.valid
    assert_matches_truth(result, doc)
    assert any("fell back to the rule extractor" in w for w in result.warnings)


def test_photo_metadata_never_reaches_the_vision_model():
    doc = make_doc("receipt", 2)
    llm = FakeLLM(page=lambda model, n: oracle_page(doc, "photo", n))
    p, transport = pipeline(LLM_ENV, llm=llm)
    result = p.extract(render(doc, "photo").data, "receipt", CLOUD)
    url = transport.requests[0].body["messages"][1]["content"][1]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert len(image.getexif()) == 0 and image.height > image.width
    assert result.source["kind"] == "jpeg" and result.valid


def test_parallel_page_reads_give_the_same_result_as_sequential_ones():
    doc = make_doc("invoice", 5)
    results = []
    for workers in ("1", "4"):
        llm = FakeLLM(page=lambda model, n: oracle_page(doc, "scan", n))
        p, _ = pipeline({**LLM_ENV, "PRIVDOC_CONCURRENCY": workers}, llm=llm)
        results.append(p.extract(render(doc, "scan").data, "invoice", CLOUD))
    assert results[0].data == results[1].data
    assert [p.reader for p in results[0].pages] == [p.reader for p in results[1].pages]


def test_links_are_fetched_safely_in_the_cloud_tier_and_the_host_recorded():
    doc = make_doc("receipt", 1)
    data = render(doc, "text").data

    class FakeFetcher:
        def fetch(self, url):
            return Fetched(url, url, data, "pdf", "application/pdf", ["docs.example.com"])

    p, _ = pipeline({"PRIVDOC_TIER": "cloud"}, fetcher=FakeFetcher())
    result = p.extract("https://docs.example.com/files/receipt.pdf?sig=abc", "receipt")
    assert result.source["downloaded_from"] == "docs.example.com"
    assert result.source["name"] == "receipt.pdf" and result.valid


def test_page_text_is_only_included_when_asked_for():
    doc = make_doc("receipt", 1)
    p, _ = pipeline()
    data = render(doc, "text").data
    assert "text" not in p.extract(data, "receipt", LOCAL).to_dict()["pages"][0]
    with_text = p.extract(data, "receipt", ExtractOptions(tier=Tier.LOCAL, include_text=True))
    assert doc.truth["merchant_name"] in with_text.to_dict()["pages"][0]["text"]


# -- cache -------------------------------------------------------------------------------------

def scanned_invoice_run(tmp_path, llm):
    doc = make_doc("invoice", 5)
    p, _ = pipeline(LLM_ENV, llm=llm, cache_dir=tmp_path)
    return doc, p.extract(render(doc, "scan").data, "invoice", CLOUD)


def test_a_second_run_is_served_from_the_cache(tmp_path):
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: oracle_page(doc, "scan", n))
    _, first = scanned_invoice_run(tmp_path, llm)
    calls = len(llm.calls)
    _, second = scanned_invoice_run(tmp_path, llm)
    assert len(llm.calls) == calls == 2
    assert second.cache_hits == 2 and second.usage.cost_usd == 0 and second.usage.llm_calls == 0
    assert second.data == first.data and [p.cached for p in second.pages] == [True, True]


def test_a_new_prompt_version_invalidates_the_cache(tmp_path, monkeypatch):
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: oracle_page(doc, "scan", n))
    scanned_invoice_run(tmp_path, llm)
    monkeypatch.setattr("privdoc.pipeline.PROMPT_VERSION", "next-version")
    _, again = scanned_invoice_run(tmp_path, llm)
    assert len(llm.calls) == 4 and again.cache_hits == 0


def test_failed_readings_are_never_cached(tmp_path):
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: 500 if n == 2 else oracle_page(doc, "scan", n))
    scanned_invoice_run(tmp_path, llm)
    before = [c.page for c in llm.calls].count(2)
    _, again = scanned_invoice_run(tmp_path, llm)
    assert [c.page for c in llm.calls].count(2) == 2 * before
    assert again.cache_hits == 1


def test_settings_that_change_answers_are_part_of_the_cache_key(tmp_path):
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: oracle_page(doc, "scan", n))
    data = render(doc, "scan").data
    for max_px in ("2000", "1600", "1600"):
        p, _ = pipeline({**LLM_ENV, "PRIVDOC_LLM_MAX_PX": max_px}, llm=llm, cache_dir=tmp_path)
        p.extract(data, "invoice", CLOUD)
    assert len(llm.calls) == 4  # 2000 px: 2 pages; 1600 px: 2 pages; 1600 again: cached


def test_a_page_number_sent_as_text_still_counts():
    doc = make_doc("invoice", 3)

    def fields(model, pages, text):
        answer = oracle_fields(doc, "text", pages)
        answer["fields"]["total"]["page"] = "2"
        return answer

    p, _ = pipeline(LLM_ENV, llm=FakeLLM(fields=fields))
    result = p.extract(render(doc, "text").data, "invoice", CLOUD)
    assert result.fields["total"].page == 2 and result.fields["total"].confidence == "high"


def test_cached_pages_do_not_report_the_original_reading_time(tmp_path):
    doc = make_doc("invoice", 5)
    llm = FakeLLM(page=lambda model, n: oracle_page(doc, "scan", n))
    scanned_invoice_run(tmp_path, llm)
    _, again = scanned_invoice_run(tmp_path, llm)
    assert all(page.cached and page.seconds == 0 for page in again.pages)
