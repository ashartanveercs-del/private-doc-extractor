import io

import pytest
from PIL import Image

from privdoc.sniff import DOCUMENT_KINDS, IMAGE_KINDS, sniff


def _image_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8, 8), 255).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt, kind", [
    ("PNG", "png"), ("JPEG", "jpeg"), ("TIFF", "tiff"), ("WEBP", "webp"), ("GIF", "gif"),
    ("BMP", "bmp"),
])
def test_images_are_recognised_by_their_bytes(fmt, kind):
    assert sniff(_image_bytes(fmt)) == kind
    assert kind in IMAGE_KINDS


def test_pdf_header_may_follow_leading_junk_within_the_first_kilobyte():
    assert sniff(b"%PDF-1.7\n...") == "pdf"
    assert sniff(b"\x00" * 300 + b"%PDF-1.4\n") == "pdf"
    assert sniff(b"\x00" * 2000 + b"%PDF-1.4\n") is None


def test_name_is_irrelevant_html_and_text_are_not_documents():
    assert sniff(b"<!doctype html><html><body>login</body></html>") is None
    assert sniff(b"just some text") is None
    assert sniff(b"") is None


def test_zip_is_recognised_but_is_not_a_document_kind():
    assert sniff(b"PK\x03\x04" + b"\x00" * 26) == "zip"
    assert "zip" not in DOCUMENT_KINDS
    assert "pdf" in DOCUMENT_KINDS
