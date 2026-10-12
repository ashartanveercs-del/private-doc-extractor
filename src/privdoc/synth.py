"""Synthetic documents with known ground truth, generated entirely in code.

Used by the tests, the examples and the benchmark, so no real document is ever needed. All
names, addresses and numbers are invented (phone numbers use the fictional 555 range).

Each document can be rendered as:
  text             a PDF with a real text layer
  scan             a PDF whose pages are images only (rotated slightly, blurred, speckled)
  searchable-scan  a scan plus an invisible text layer holding only the printed labels, the
                   way a copier's OCR layer misses handwritten entries
  mixed            first page with a text layer, later pages scanned
  photo            a sideways JPEG with an EXIF orientation tag and GPS metadata
"""

from __future__ import annotations

import datetime as dt
import functools
import io
import random
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .schema import load_schema

KINDS = ("invoice", "receipt", "tax_form")
VARIANTS = ("text", "scan", "searchable-scan", "mixed", "photo")
PAGE_W, PAGE_H = 612.0, 792.0  # US Letter in points
LEFT, TOP = 72.0, 60.0
SCAN_DPI = 150

VENDORS = ["Cedar Lane Plumbing", "Northwind Paper Co.", "Bluebird Bakery",
           "Harbor View Print Shop", "Maple Street Florist", "Summit Bike Repair",
           "Copperleaf Catering", "Riverbend Landscaping"]
STORES = ["Corner Bean Cafe", "Green Basket Market", "Sunrise Deli", "Lakeside Hardware",
          "Pine Hill Pharmacy", "Golden Crust Pizza"]
PEOPLE = ["Jordan Lee", "Sam Rivera", "Alex Morgan", "Taylor Brooks", "Casey Nguyen",
          "Riley Patel", "Morgan Diaz", "Jamie Chen"]
STREETS = ["Example Street", "Sample Avenue", "Placeholder Road", "Demo Boulevard",
           "Specimen Lane"]
CITIES = ["Brightwater", "Riverton", "Lakeside", "Fairview", "Oakdale"]
STATES = ["CA", "NY", "TX", "WA", "IL", "OR", "CO", "MA"]
ITEMS = ["Copy paper (case)", "Toner cartridge", "Desk lamp", "USB-C cable",
         "Notebook pack", "Labor - 2 hours", "Pipe fitting kit", "Delivery fee",
         "Copper elbow", "Tune-up service", "Seasonal bouquet", "Catering tray"]
RECEIPT_ITEMS = ["Coffee", "Bagel", "Orange juice", "Sandwich", "Garden salad", "Muffin",
                 "Green tea", "Soup of the day", "Batteries", "Paint brush"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
DATE_STYLES = ("iso", "us", "long", "short")
MONEY_STYLES = ("dollar", "plain", "code")


# -- layout ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    cells: tuple[tuple[float, str], ...]
    size: float = 11.0
    bold: bool = False
    gap: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(text for _, text in self.cells)


def _line(text: str, size: float = 11.0, bold: bool = False, gap: float = 0.0) -> Line:
    return Line(((LEFT, text),), size, bold, gap)


@dataclass(frozen=True)
class TextRun:
    x: float
    y: float  # baseline, in points from the top of the page
    text: str
    size: float = 11.0
    bold: bool = False
    invisible: bool = False


@dataclass
class PdfPage:
    runs: list[TextRun] = field(default_factory=list)
    image: Image.Image | None = None  # drawn full-page, underneath the text
    width: float = PAGE_W
    height: float = PAGE_H
    drawing: bytes = b""  # raw content-stream operators, e.g. b"72 600 200 40 re S"
    image_over_text: bool = False  # paint the image last, hiding the text (as some scanners do)


def _layout(lines: list[Line]) -> tuple[list[TextRun], float]:
    runs, y = [], TOP
    for line in lines:
        y += line.gap + line.size
        runs.extend(TextRun(x, y, text, line.size, line.bold) for x, text in line.cells)
        y += line.size * 0.6
    return runs, y


# -- a minimal PDF writer ----------------------------------------------------------------------

def _pdf_string(text: str) -> bytes:
    raw = text.encode("cp1252", errors="replace")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _stream(header: bytes, payload: bytes) -> bytes:
    return header[:-2] + b" /Length %d >>\nstream\n" % len(payload) + payload + b"\nendstream"


def write_pdf(pages: list[PdfPage]) -> bytes:
    """Write pages with Helvetica text runs and optional full-page grayscale JPEG images."""
    bodies: list[bytes | None] = [None, None]  # 1: catalog, 2: page tree
    def add(body: bytes) -> int:
        bodies.append(body)
        return len(bodies)

    regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                  b"/Encoding /WinAnsiEncoding >>")
    bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
               b"/Encoding /WinAnsiEncoding >>")
    kids = []
    for page in pages:
        ops, xobjects, paint_image = [], b"", b""
        if page.image is not None:
            image = page.image.convert("L")
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=75)
            img = add(_stream(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d "
                              b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode >>"
                              % image.size, buf.getvalue()))
            xobjects = b" /XObject << /Im1 %d 0 R >>" % img
            paint_image = b"q %.2f 0 0 %.2f 0 0 cm /Im1 Do Q" % (page.width, page.height)
        if paint_image and not page.image_over_text:
            ops.append(paint_image)
        if page.drawing:
            ops.append(page.drawing)
        for run in page.runs:
            mode = b"3 Tr " if run.invisible else b""
            ops.append(b"BT %s/F%d %.1f Tf %.2f %.2f Td %s Tj ET" % (
                mode, 2 if run.bold else 1, run.size, run.x, page.height - run.y,
                _pdf_string(run.text)))
        if paint_image and page.image_over_text:
            ops.append(paint_image)
        content = add(_stream(b"<< >>", b"\n".join(ops)))
        kids.append(add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /Font << "
            b"/F1 %d 0 R /F2 %d 0 R >>%s >> /Contents %d 0 R >>"
            % (page.width, page.height, regular, bold, xobjects, content)))
    bodies[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    bodies[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids), len(kids))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(bodies) + 1))
    out.write(b"".join(b"%010d 00000 n \n" % offset for offset in offsets))
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
              % (len(bodies) + 1, xref))
    return out.getvalue()


# -- rasterising (the "scanner") ---------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=px)


def _rasterize(runs: list[TextRun], width: float, height: float, dpi: int) -> Image.Image:
    scale = dpi / 72
    image = Image.new("L", (round(width * scale), round(height * scale)), 255)
    draw = ImageDraw.Draw(image)
    for run in runs:
        if run.invisible:
            continue
        draw.text((run.x * scale, run.y * scale), run.text, fill=25, anchor="ls",
                  font=_font(round(run.size * scale)), stroke_width=1 if run.bold else 0,
                  stroke_fill=25)
    return image


def _degrade(image: Image.Image, rng: random.Random, max_angle: float) -> Image.Image:
    image = image.rotate(rng.uniform(-max_angle, max_angle), resample=Image.Resampling.BICUBIC,
                         fillcolor=255)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(width * height // 1500):
        draw.point((rng.randrange(width), rng.randrange(height)), fill=rng.randint(0, 140))
    return image.filter(ImageFilter.GaussianBlur(0.6))


def _scan(lines: list[Line], rng: random.Random, dpi: int = SCAN_DPI) -> Image.Image:
    runs, _ = _layout(lines)
    return _degrade(_rasterize(runs, PAGE_W, PAGE_H, dpi), rng, max_angle=0.6)


# -- documents ---------------------------------------------------------------------------------

@dataclass
class SynthDoc:
    kind: str
    seed: int
    truth: dict[str, Any]
    fields: list[str]
    pages: list[list[Line]]

    @property
    def absent(self) -> list[str]:
        return [name for name in self.fields if name not in self.truth]

    @property
    def text(self) -> str:
        return "\n".join(line.text for page in self.pages for line in page)


@dataclass
class Rendered:
    data: bytes
    kind: str  # "pdf" | "jpeg"
    name: str
    pages: int


def _date(day: dt.date, style: str) -> str:
    if style == "iso":
        return day.isoformat()
    if style == "us":
        return f"{day.month:02d}/{day.day:02d}/{day.year}"
    if style == "long":
        return f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
    return f"{day.day} {MONTHS[day.month - 1][:3]} {day.year}"


def _money(value: float, style: str) -> str:
    text = f"{value:,.2f}"
    return {"dollar": f"${text}", "plain": text, "code": f"USD {text}"}[style]


def _random_date(rng: random.Random) -> dt.date:
    return dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(600))


def _address(rng: random.Random) -> str:
    return (f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}, "
            f"{rng.choice(STATES)} {rng.randint(10000, 99999)}")


def _invoice(rng: random.Random) -> tuple[dict[str, Any], list[list[Line]]]:
    vendor, customer = rng.choice(VENDORS), rng.choice(PEOPLE)
    ds, ms = rng.choice(DATE_STYLES), rng.choice(MONEY_STYLES)
    issued = _random_date(rng)
    truth: dict[str, Any] = {"vendor_name": vendor,
                             "invoice_number": f"INV-{rng.randint(1000, 99999)}",
                             "invoice_date": issued.isoformat(), "customer_name": customer}
    page1 = [_line(vendor, 16, True), _line(_address(rng)),
             _line(f"Phone: +1 555 000 {rng.randint(0, 9999):04d}"),
             _line("INVOICE", 14, True, gap=10), _line(f"Invoice #: {truth['invoice_number']}"),
             _line(f"Invoice date: {_date(issued, ds)}")]
    if rng.random() < 0.8:
        due = issued + dt.timedelta(days=rng.choice((14, 30, 45)))
        truth["due_date"] = due.isoformat()
        page1.append(_line(f"Due date: {_date(due, ds)}"))
    if rng.random() < 0.5:
        truth["po_number"] = f"PO-{rng.randint(10000, 99999)}"
        page1.append(_line(f"PO number: {truth['po_number']}"))
    page1.append(_line(f"Bill to: {customer}", gap=8))
    items = []
    for description in rng.sample(ITEMS, rng.randint(2, 4)):
        quantity, unit = rng.randint(1, 5), rng.randint(250, 18000) / 100
        items.append({"description": description, "quantity": quantity, "unit_price": unit,
                      "amount": round(quantity * unit, 2)})
    subtotal = round(sum(i["amount"] for i in items), 2)
    truth.update(line_items=items, subtotal=subtotal)
    page2 = [_line("Totals", 13, True), _line(f"Subtotal: {_money(subtotal, ms)}")]
    total = subtotal
    if rng.random() < 0.7:
        tax = round(subtotal * rng.choice((0.05, 0.0625, 0.08, 0.0825)), 2)
        truth["tax"] = tax
        total = round(subtotal + tax, 2)
        page2.append(_line(f"Tax: {_money(tax, ms)}"))
    truth["total"] = total
    page2 += [_line(f"Total due: {_money(total, ms)}", bold=True),
              _line(f"Payable to: {vendor}", gap=10), _line("Thank you for your business.")]
    # Real invoices print "Unit price | Qty" as often as "Qty | Unit price". The order is drawn
    # last, so every other draw (and so every value) is the same whichever order is printed.
    page1 += _item_table(items, rng.sample(ITEM_COLUMNS, len(ITEM_COLUMNS)))
    return truth, [page1, page2]


ITEM_COLUMNS = (("quantity", "Qty"), ("unit_price", "Unit price"), ("amount", "Amount"))
_COLUMN_X = (340.0, 400.0, 490.0)


def _item_table(items: list[dict[str, Any]],
                columns: list[tuple[str, str]]) -> list[Line]:
    """The line-item table, with its number columns printed in the given order."""
    def cell(value: Any) -> str:
        return str(value) if isinstance(value, int) else f"{value:.2f}"

    header = Line(((LEFT, "Description"),
                   *((x, label) for x, (_, label) in zip(_COLUMN_X, columns, strict=True))),
                  bold=True, gap=10)
    return [header] + [Line(((LEFT, item["description"]),
                             *((x, cell(item[name]))
                               for x, (name, _) in zip(_COLUMN_X, columns, strict=True))))
                       for item in items]


def _receipt(rng: random.Random) -> tuple[dict[str, Any], list[list[Line]]]:
    store = rng.choice(STORES)
    ds, ms = rng.choice(DATE_STYLES), rng.choice(MONEY_STYLES)
    day = _random_date(rng)
    truth: dict[str, Any] = {"merchant_name": store, "date": day.isoformat()}
    lines = [_line(store, 14, True), _line(f"Store: {store}"), _line(_address(rng))]
    date_cell = (LEFT, f"Date: {_date(day, ds)}")
    if rng.random() < 0.7:
        truth["time"] = f"{rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}"
        lines.append(Line((date_cell, (330.0, f"Time: {truth['time']}"))))
    else:
        lines.append(Line((date_cell,)))
    if rng.random() < 0.6:
        truth["receipt_number"] = f"R-{rng.randint(100000, 999999)}"
        lines.append(_line(f"Receipt #: {truth['receipt_number']}"))
    subtotal = 0.0
    for index, name in enumerate(rng.sample(RECEIPT_ITEMS, rng.randint(1, 5))):
        price = rng.randint(150, 1500) / 100
        subtotal += price
        lines.append(Line(((LEFT, name), (400.0, f"{price:.2f}")), gap=8 if index == 0 else 0))
    subtotal = round(subtotal, 2)
    truth["subtotal"] = subtotal
    lines.append(_line(f"Subtotal: {_money(subtotal, ms)}", gap=8))
    total = subtotal
    if rng.random() < 0.8:
        truth["tax"] = round(subtotal * 0.08, 2)
        total += truth["tax"]
        lines.append(_line(f"Tax: {_money(truth['tax'], ms)}"))
    if rng.random() < 0.3:
        truth["tip"] = float(rng.randint(1, 10))
        total += truth["tip"]
        lines.append(_line(f"Tip: {_money(truth['tip'], ms)}"))
    truth["total"] = round(total, 2)
    lines.append(_line(f"Total: {_money(truth['total'], ms)}", bold=True))
    if rng.random() < 0.6:
        truth["payment_method"] = "card"
        truth["card_last4"] = f"{rng.randint(0, 9999):04d}"
        lines += [_line("Payment: Card", gap=6), _line(f"Card ending: {truth['card_last4']}")]
    else:
        truth["payment_method"] = "cash"
        lines.append(_line("Payment: Cash", gap=6))
    lines.append(_line("Thank you!", gap=8))
    return truth, [lines]


def _tax_form(rng: random.Random) -> tuple[dict[str, Any], list[list[Line]]]:
    ms = rng.choice(MONEY_STYLES)
    truth: dict[str, Any] = {
        "tax_year": rng.choice((2024, 2025)), "payer_name": rng.choice(VENDORS),
        "recipient_name": rng.choice(PEOPLE), "compensation": rng.randint(60000, 9000000) / 100,
        "corrected": rng.random() < 0.25}
    lines = [_line("CONTRACTOR PAYMENT STATEMENT", 14, True),
             _line(f"Tax year: {truth['tax_year']}", gap=6),
             _line(f"Corrected: {'[x]' if truth['corrected'] else '[ ]'}"),
             _line(f"Payer name: {truth['payer_name']}", gap=10)]
    if rng.random() < 0.8:
        truth["payer_tin"] = f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
        lines.append(_line(f"Payer TIN: {truth['payer_tin']}"))
    lines.append(_line(f"Recipient name: {truth['recipient_name']}", gap=10))
    if rng.random() < 0.8:
        truth["recipient_tin"] = f"***-**-{rng.randint(0, 9999):04d}"
        lines.append(_line(f"Recipient TIN: {truth['recipient_tin']}"))
    lines.append(_line(f"Nonemployee compensation: {_money(truth['compensation'], ms)}", gap=10))
    if rng.random() < 0.5:
        truth["federal_tax_withheld"] = round(truth["compensation"] * 0.1, 2)
        lines.append(_line(f"Federal income tax withheld: "
                           f"{_money(truth['federal_tax_withheld'], ms)}"))
    if rng.random() < 0.6:
        truth["state"] = rng.choice(STATES)
        lines.append(_line(f"State: {truth['state']}"))
    return truth, [lines]


_TEMPLATES = {"invoice": _invoice, "receipt": _receipt, "tax_form": _tax_form}


def make_doc(kind: str, seed: int) -> SynthDoc:
    rng = random.Random(f"{kind}:{seed}")
    truth, pages = _TEMPLATES[kind](rng)
    return SynthDoc(kind, seed, truth, [f.name for f in load_schema(kind).fields], pages)


def _labels_only(runs: list[TextRun]) -> list[TextRun]:
    return [TextRun(r.x, r.y, r.text.split(":")[0] + ":", r.size, r.bold, invisible=True)
            for r in runs if ":" in r.text]


def _photo(doc: SynthDoc, rng: random.Random) -> bytes:
    runs, bottom = _layout([line for page in doc.pages for line in page])
    image = _degrade(_rasterize(runs, PAGE_W, max(PAGE_H, bottom + 60), SCAN_DPI), rng, 1.5)
    stored = image.rotate(90, expand=True)  # held sideways; EXIF orientation 6 turns it back
    exif = Image.Exif()
    exif[0x0112] = 6
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (12.0, 34.0, 56.0), 3: "E", 4: (65.0, 43.0, 21.0)})
    buf = io.BytesIO()
    stored.save(buf, format="JPEG", quality=80, exif=exif.tobytes())
    return buf.getvalue()


def variant_pages(doc: SynthDoc, variant: str) -> list[list[Line]]:
    """The lines on each page of a rendered variant."""
    if variant == "photo":
        return [[line for page in doc.pages for line in page]]
    if variant == "mixed" and len(doc.pages) == 1:
        half = (len(doc.pages[0]) + 1) // 2
        return [doc.pages[0][:half], doc.pages[0][half:]]
    return doc.pages


def render(doc: SynthDoc, variant: str) -> Rendered:
    rng = random.Random(f"{doc.kind}:{doc.seed}:{variant}")
    name = f"{doc.kind}-{doc.seed}-{variant}"
    if variant == "photo":
        return Rendered(_photo(doc, rng), "jpeg", f"{name}.jpg", 1)
    out = []
    for index, lines in enumerate(variant_pages(doc, variant)):
        runs, _ = _layout(lines)
        if variant == "text" or (variant == "mixed" and index == 0):
            out.append(PdfPage(runs=runs))
        elif variant in ("scan", "mixed"):
            out.append(PdfPage(image=_scan(lines, rng)))
        elif variant == "searchable-scan":
            out.append(PdfPage(runs=_labels_only(runs), image=_scan(lines, rng)))
        else:
            raise ValueError(f"unknown variant {variant!r}")
    return Rendered(write_pdf(out), "pdf", f"{name}.pdf", len(out))
