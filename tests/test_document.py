import hashlib
import io

import pytest
from PIL import Image

from privdoc.document import (
    MAX_RENDER_PIXELS,
    encode_png,
    open_document,
    parse_page_spec,
)
from privdoc.errors import DocumentError, UsageError
from privdoc.routing import PageFacts, Route, RoutingRules, route_page
from privdoc.synth import PdfPage, TextRun, make_doc, render, write_pdf

RULES = RoutingRules()
TEXT = "Invoice #: INV-1042  Invoice date: 2026-03-14  Total due: $1,250.00"


# -- routing rules (pure) ----------------------------------------------------------------------

def test_a_page_with_a_real_text_layer_is_read_for_free():
    route, reason = route_page(PageFacts(1, TEXT), RULES)
    assert route is Route.TEXT and "characters" in reason


def test_a_searchable_scan_is_ocred_even_though_it_has_text():
    route, reason = route_page(PageFacts(1, TEXT, image_cover=0.95, image_count=1), RULES)
    assert route is Route.OCR and "95%" in reason


def test_a_typed_page_with_a_small_logo_stays_on_the_text_layer():
    assert route_page(PageFacts(1, TEXT, image_cover=0.05, image_count=1), RULES)[0] is Route.TEXT


def test_little_text_plus_an_image_is_ocred():
    assert route_page(PageFacts(1, "Page 2", image_cover=0.3, image_count=1), RULES)[0] is Route.OCR


def test_vector_only_pages_are_ocred_not_mistaken_for_blank():
    route, reason = route_page(PageFacts(1, "", has_vector_ink=True), RULES)
    assert route is Route.OCR and "drawing" in reason


def test_a_page_with_nothing_on_it_is_blank():
    assert route_page(PageFacts(1, "  "), RULES)[0] is Route.BLANK


def test_a_short_text_page_with_nothing_else_is_still_text():
    assert route_page(PageFacts(1, "Notes:"), RULES)[0] is Route.TEXT


def test_force_ocr_and_thresholds_are_configurable():
    assert route_page(PageFacts(1, TEXT), RoutingRules(force_ocr=True))[0] is Route.OCR
    no_cover_rule = RoutingRules(scan_cover=1.01)
    assert route_page(PageFacts(1, TEXT, image_cover=1.0, image_count=1),
                      no_cover_rule)[0] is Route.TEXT


# -- opening documents -------------------------------------------------------------------------

def test_text_pdf_facts():
    rendered = render(make_doc("invoice", 1), "text")
    with open_document(rendered.data, "invoice.pdf") as doc:
        assert (doc.kind, doc.page_count, doc.name) == ("pdf", 2, "invoice.pdf")
        assert doc.sha256 == hashlib.sha256(rendered.data).hexdigest()
        facts = doc.facts(1)
        assert "Invoice #:" in facts.text and facts.image_cover == 0


def test_scanned_pdf_facts_and_render():
    with open_document(render(make_doc("receipt", 1), "scan").data) as doc:
        facts = doc.facts(1)
        assert facts.text == "" and facts.image_cover == pytest.approx(1.0)
        image = doc.render(1, dpi=100)
        assert image.mode == "L" and image.size == (850, 1100)


def test_blank_and_vector_only_pages():
    data = write_pdf([PdfPage(), PdfPage(drawing=b"0 G 2 w 72 500 300 120 re S")])
    with open_document(data) as doc:
        assert route_page(doc.facts(1), RULES)[0] is Route.BLANK
        assert doc.facts(2).has_vector_ink and route_page(doc.facts(2), RULES)[0] is Route.OCR


def test_huge_pages_are_rendered_within_the_pixel_budget():
    data = write_pdf([PdfPage(runs=[TextRun(72, 100, "big")], width=5000, height=5000)])
    with open_document(data) as doc:
        image = doc.render(1, dpi=300)
    assert image.width * image.height <= MAX_RENDER_PIXELS


def test_image_inputs_are_one_ocr_page_per_frame():
    frames = [Image.new("L", (40, 60), shade) for shade in (0, 128, 255)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    with open_document(buf.getvalue(), "fax.tif") as doc:
        assert (doc.kind, doc.page_count) == ("tiff", 3)
        assert doc.facts(2).image_cover == 1.0
        assert doc.render(2, dpi=300).getpixel((5, 5)) == 128


def test_photos_are_turned_upright_and_lose_their_metadata_before_encoding():
    photo = render(make_doc("receipt", 2), "photo").data
    assert Image.open(io.BytesIO(photo)).getexif().get_ifd(0x8825)  # it carries GPS data
    with open_document(photo, "receipt.jpg") as doc:
        image = doc.render(1, dpi=300)
    assert image.height > image.width
    png = encode_png(image)
    assert png.startswith(b"\x89PNG") and b"eXIf" not in png and b"Exif" not in png
    assert len(Image.open(io.BytesIO(png)).getexif()) == 0


@pytest.mark.parametrize("data, message", [
    (b"%PDF-1.4\nthis is not really a pdf", "could not open"),
    (b"<html>login</html>", "not a supported"),
    (b"PK\x03\x04" + b"\x00" * 40, "zip"),
])
def test_unreadable_inputs_fail_with_a_clear_message(data, message):
    with pytest.raises(DocumentError, match=message):
        open_document(data)


def test_size_cap():
    with pytest.raises(DocumentError, match="larger"):
        open_document(b"%PDF-" + b"0" * 2000, max_bytes=1000)


@pytest.mark.parametrize("spec, total, expected", [
    (None, 3, [1, 2, 3]), ("", 3, [1, 2, 3]), ("1-3,5", 6, [1, 2, 3, 5]), ("3-", 5, [3, 4, 5]),
    ("2, 2, 1", 4, [1, 2]),
])
def test_page_specs(spec, total, expected):
    assert parse_page_spec(spec, total) == expected


@pytest.mark.parametrize("spec", ["0", "7", "2-1", "a", "1-x", "-"])
def test_bad_page_specs(spec):
    with pytest.raises(UsageError):
        parse_page_spec(spec, 5)
