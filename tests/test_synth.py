import io

import pdfplumber
import pytest
from PIL import Image, ImageOps

from privdoc.schema import load_schema
from privdoc.synth import KINDS, VARIANTS, Line, PdfPage, TextRun, make_doc, render, write_pdf


def pdf_pages(data: bytes):
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [(p.extract_text() or "", [(i["x1"] - i["x0"]) * (i["bottom"] - i["top"])
                                          / (p.width * p.height) for i in p.images])
                for p in pdf.pages]


def test_kinds_and_variants():
    assert KINDS == ("invoice", "receipt", "tax_form")
    assert VARIANTS == ("text", "scan", "searchable-scan", "mixed", "photo")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(6))
def test_ground_truth_is_always_valid_against_the_builtin_schema(kind, seed):
    doc = make_doc(kind, seed)
    schema = load_schema(kind)
    assert schema.validate(doc.truth) == []
    assert set(doc.fields) == {f.name for f in schema.fields}
    assert set(doc.absent) == set(doc.fields) - set(doc.truth)


def test_documents_are_deterministic_per_seed():
    assert make_doc("invoice", 3).truth == make_doc("invoice", 3).truth
    assert make_doc("invoice", 3).text == make_doc("invoice", 3).text
    assert render(make_doc("receipt", 3), "scan").data == render(make_doc("receipt", 3),
                                                                  "scan").data
    assert len({make_doc("invoice", s).truth["invoice_number"] for s in range(10)}) > 5


def test_invoices_print_their_line_item_columns_in_varied_orders():
    headers = {next(line.text for line in make_doc("invoice", seed).pages[0]
                    if line.text.startswith("Description")) for seed in range(12)}
    assert "Description Qty Unit price Amount" in headers and len(headers) > 2


def test_absent_fields_are_really_absent_from_the_page():
    docs = [make_doc("invoice", s) for s in range(20)]
    without_po = [d for d in docs if "po_number" in d.absent]
    with_po = [d for d in docs if "po_number" in d.truth]
    assert without_po and with_po
    assert all("PO number" not in d.text for d in without_po)
    assert all(d.truth["po_number"] in d.text for d in with_po)


def test_text_variant_has_a_real_text_layer_with_the_values():
    doc = make_doc("invoice", 1)
    pages = pdf_pages(render(doc, "text").data)
    assert len(pages) == 2
    text = "\n".join(t for t, _ in pages)
    assert doc.truth["invoice_number"] in text and doc.truth["vendor_name"] in text
    assert all(covers == [] for _, covers in pages)


def test_scan_variant_is_images_only():
    pages = pdf_pages(render(make_doc("invoice", 1), "scan").data)
    assert [(t, [round(c, 2) for c in covers]) for t, covers in pages] == [("", [1.0]),
                                                                           ("", [1.0])]


def test_searchable_scan_has_labels_in_its_text_layer_but_not_the_values():
    doc = make_doc("tax_form", 2)
    (text, covers), = pdf_pages(render(doc, "searchable-scan").data)
    assert "Payer name:" in text and covers == [pytest.approx(1.0)]
    assert doc.truth["payer_name"] not in text


def test_mixed_variant_has_a_text_page_and_a_scanned_page():
    doc = make_doc("receipt", 4)
    pages = pdf_pages(render(doc, "mixed").data)
    assert len(pages) == 2
    assert pages[0][0] and pages[0][1] == []
    assert pages[1][0] == "" and pages[1][1] == [pytest.approx(1.0)]


def test_photo_variant_is_a_sideways_jpeg_with_gps_metadata():
    rendered = render(make_doc("receipt", 5), "photo")
    assert rendered.kind == "jpeg"
    image = Image.open(io.BytesIO(rendered.data))
    exif = image.getexif()
    assert exif.get(0x0112) == 6 and exif.get_ifd(0x8825)
    upright = ImageOps.exif_transpose(image)
    assert upright.height > upright.width


def test_pdf_writer_escapes_text_and_supports_invisible_runs():
    data = write_pdf([PdfPage(runs=[TextRun(72, 100, "Total (USD): 5\\6"),
                                    TextRun(72, 130, "hidden label", invisible=True)])])
    (text, _), = pdf_pages(data)
    assert text.splitlines() == ["Total (USD): 5\\6", "hidden label"]


def test_line_text_joins_its_cells():
    line = Line(((72.0, "Desk lamp"), (340.0, "2"), (400.0, "18.00")))
    assert line.text == "Desk lamp 2 18.00"
