"""Open a PDF or an image, describe each page, and render pages for OCR.

PDFs are parsed with pdfplumber (pdfminer.six for text and layout, pypdfium2 for rendering);
images with Pillow. The file type comes from its bytes, never its name. Images are turned
upright from their EXIF orientation, and every rendered page leaves here as a fresh image with
no metadata, so a phone photo's GPS position is never sent anywhere.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import threading
import warnings
from typing import Protocol

import pdfplumber
from pdfminer.pdfdevice import PDFTextDevice
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.utils import apply_matrix_rect
from PIL import Image, ImageOps

from .errors import DocumentError, UsageError
from .routing import PageFacts
from .sniff import IMAGE_KINDS, sniff

MAX_RENDER_PIXELS = 40_000_000
_INVISIBLE_MODES = frozenset({3, 7})  # text render modes that paint nothing (3: invisible, 7: clip)


class _PaintedText(PDFTextDevice):
    """Replays a page's drawing operations to see which characters really end up visible.

    pdfplumber reports every character of the text layer, painted or not. A copier's searchable
    PDF draws its OCR text in an invisible render mode over the page image, and some scanners
    hide that text under the image instead; typed text drawn over a background image is
    visible. A character counts as visible when it is drawn in a painting mode and no image is
    drawn over its position afterwards.
    """

    def __init__(self, rsrcmgr) -> None:
        super().__init__(rsrcmgr)
        self._mode = 0
        self.chars: list[tuple[float, float, bool]] = []  # origin x, y; painted?
        self.images: list[tuple[int, tuple[float, float, float, float]]] = []  # after char n

    def render_string(self, textstate, seq, ncs, graphicstate) -> None:
        self._mode = textstate.render
        super().render_string(textstate, seq, ncs, graphicstate)

    def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs,
                    graphicstate) -> float:
        try:
            text = font.to_unichr(cid)
        except PDFUnicodeNotDefined:
            text = "?"
        if text.strip():
            self.chars.append((matrix[4], matrix[5], self._mode not in _INVISIBLE_MODES))
        return font.char_width(cid) * fontsize * scaling  # the advance, as pdfminer computes it

    def render_image(self, name, stream) -> None:
        self.images.append((len(self.chars), apply_matrix_rect(self.ctm, (0, 0, 1, 1))))

    def visible_share(self) -> float | None:
        if not self.chars:
            return None
        visible = 0
        for index, (x, y, painted) in enumerate(self.chars):
            covered = any(index < drawn_after and x0 <= x <= x1 and y0 <= y <= y1
                          for drawn_after, (x0, y0, x1, y1) in self.images)
            visible += painted and not covered
        return visible / len(self.chars)


class Document(Protocol):
    name: str
    kind: str
    sha256: str
    page_count: int

    def facts(self, number: int) -> PageFacts: ...

    def render(self, number: int, dpi: int) -> Image.Image: ...

    def close(self) -> None: ...

    def __enter__(self) -> Document: ...

    def __exit__(self, *exc: object) -> None: ...


class _Base:
    def __init__(self, data: bytes, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.page_count = 0
        self._lock = threading.Lock()  # pdfium and Pillow objects are not thread-safe
        self._facts: dict[int, PageFacts] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check(self, number: int) -> None:
        if not 1 <= number <= self.page_count:
            raise DocumentError(f"page {number} does not exist")


class PdfDocument(_Base):
    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data, name, "pdf")
        try:
            self._pdf = pdfplumber.open(io.BytesIO(data))
            self.page_count = len(self._pdf.pages)
        except Exception:  # pdfminer raises many types for damaged or encrypted files
            raise DocumentError("could not open the PDF: it is damaged or "
                                "password-protected") from None
        if self.page_count == 0:
            raise DocumentError("the PDF has no pages")

    def facts(self, number: int) -> PageFacts:
        self._check(number)
        with self._lock:
            if number not in self._facts:
                page = self._pdf.pages[number - 1]
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                area = float(page.width * page.height) or 1.0
                covers = []
                for image in page.images:
                    x0, x1 = max(0.0, float(image["x0"])), min(float(page.width),
                                                               float(image["x1"]))
                    top, bottom = max(0.0, float(image["top"])), min(float(page.height),
                                                                     float(image["bottom"]))
                    covers.append(max(0.0, x1 - x0) * max(0.0, bottom - top) / area)
                visible = self._visible_share(page) if covers and text.strip() else None
                self._facts[number] = PageFacts(
                    number=number, text=text, image_cover=round(max(covers, default=0.0), 4),
                    image_count=len(covers),
                    has_vector_ink=bool(page.rects or page.lines or page.curves),
                    visible_text_share=visible)
                page.close()
            return self._facts[number]

    def _visible_share(self, page) -> float | None:
        """How much of a page's text is painted on top of its images (pages with both only)."""
        device = _PaintedText(self._pdf.rsrcmgr)
        try:
            PDFPageInterpreter(self._pdf.rsrcmgr, device).process_page(page.page_obj)
        except Exception:  # a damaged content stream: unknown, so the page is routed as a scan
            return None
        return device.visible_share()

    def render(self, number: int, dpi: int) -> Image.Image:
        self._check(number)
        with self._lock:
            page = self._pdf.pages[number - 1]
            square_inches = (float(page.width) / 72) * (float(page.height) / 72)
            dpi = min(dpi, int(math.sqrt(MAX_RENDER_PIXELS / square_inches)))
            try:
                image = page.to_image(resolution=dpi, antialias=True).original
            except Exception:
                raise DocumentError(f"could not render page {number}") from None
            finally:
                page.close()
        return _fresh(image)

    def close(self) -> None:
        self._pdf.close()


class ImageDocument(_Base):
    def __init__(self, data: bytes, name: str, kind: str) -> None:
        super().__init__(data, name, kind)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                self._image = Image.open(io.BytesIO(data))
                width, height = self._image.size
        except (OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning):
            raise DocumentError("could not open the image: it is damaged or too large") from None
        if width * height > MAX_RENDER_PIXELS:
            raise DocumentError("the image is too large to process")
        self.page_count = getattr(self._image, "n_frames", 1)

    def facts(self, number: int) -> PageFacts:
        self._check(number)
        return PageFacts(number=number, text="", image_cover=1.0, image_count=1)

    def render(self, number: int, dpi: int) -> Image.Image:
        self._check(number)
        with self._lock:
            try:
                self._image.seek(number - 1)
                upright = ImageOps.exif_transpose(self._image)
            except (OSError, ValueError, EOFError):
                raise DocumentError(f"could not decode image frame {number}") from None
        return _fresh(upright)

    def close(self) -> None:
        self._image.close()


def _fresh(image: Image.Image) -> Image.Image:
    """Grayscale copy with no metadata (EXIF, GPS, ICC, comments) attached."""
    clean = image.convert("L")
    clean.info = {}
    return clean


def encode_png(image: Image.Image) -> bytes:
    clean = image.copy()
    clean.info = {}
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return buf.getvalue()


def downscale(image: Image.Image, max_px: int) -> Image.Image:
    if max(image.size) <= max_px:
        return image
    small = image.copy()
    small.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return small


def open_document(data: bytes, name: str = "document", *,
                  max_bytes: int | None = None) -> PdfDocument | ImageDocument:
    if max_bytes is not None and len(data) > max_bytes:
        raise DocumentError(f"the file is larger than the {max_bytes / 1048576:g} MB limit")
    kind = sniff(data)
    if kind == "pdf":
        return PdfDocument(data, name)
    if kind in IMAGE_KINDS:
        return ImageDocument(data, name, kind)
    if kind == "zip":
        raise DocumentError("this is a zip archive (Office files are zip archives too); only "
                            "PDFs and images are supported")
    raise DocumentError("not a supported document: expected a PDF or an image (PNG, JPEG, "
                        "TIFF, WEBP, GIF or BMP)")


_RANGE = re.compile(r"^(\d+)(?:\s*-\s*(\d*))?$")


def parse_page_spec(spec: str | None, total: int) -> list[int]:
    """'1-3,5' -> [1, 2, 3, 5]; '3-' -> 3..total; empty -> every page."""
    if not spec or not spec.strip():
        return list(range(1, total + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        match = _RANGE.match(part.strip())
        if not match:
            raise UsageError(f"bad page range {part.strip()!r}; use e.g. 1-3,5 or 4-")
        start = int(match.group(1))
        if match.group(2) is None:
            end = start
        else:
            end = int(match.group(2)) if match.group(2) else total
        if start < 1 or end < start or end > total:
            raise UsageError(f"page range {part.strip()!r} is outside 1-{total}")
        pages.update(range(start, end + 1))
    return sorted(pages)
