"""Runs the real tesseract binary. Skipped where it is not installed; CI installs it."""

import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from privdoc.config import Settings
from privdoc.engines.tesseract import TesseractReader
from privdoc.pipeline import ExtractOptions, Pipeline
from privdoc.privacy import Tier
from privdoc.schema import load_schema
from privdoc.synth import make_doc, render

pytestmark = [pytest.mark.tesseract,
              pytest.mark.skipif(shutil.which("tesseract") is None,
                                 reason="tesseract is not installed")]


def big_text_image() -> Image.Image:
    image = Image.new("L", (1400, 300), 255)
    ImageDraw.Draw(image).text((40, 90), "INVOICE 12345", fill=0,
                               font=ImageFont.load_default(size=80))
    return image


@pytest.mark.parametrize("reader", [TesseractReader(),
                                    TesseractReader(psm=6, dpi=400, clean=True)])
def test_real_tesseract_reads_clean_text(reader):
    reading = reader.read(big_text_image(), 1, 1, load_schema("invoice"))
    assert "INVOICE" in reading.text.upper()
    assert reading.legibility in ("good", "fair")


def test_real_tesseract_inside_the_local_pipeline():
    doc = make_doc("receipt", 1)
    pipeline = Pipeline(Settings.from_env({"PRIVDOC_CACHE": "off"}))
    result = pipeline.extract(render(doc, "scan").data, "receipt",
                              ExtractOptions(tier=Tier.LOCAL, include_text=True))
    page = result.pages[0]
    assert (page.route, page.status) == ("ocr", "read")
    assert page.reader.startswith("tesseract:") and page.text.strip()
    assert result.data_sent_to == []
