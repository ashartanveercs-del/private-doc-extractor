"""Per-page routing: read the text layer for free, OCR only what is really an image.

The rules, in order:

1. ``force_ocr``                                   -> OCR
2. one image covers most of the page (>= 60 %)    -> OCR. This catches "searchable" scans: a
   copier adds an invisible OCR layer that holds the printed labels but not what was
   handwritten or stamped, so trusting that layer silently drops the values.
3. a real text layer (>= 30 characters)            -> text layer (free, exact)
4. little or no text, but images                   -> OCR
5. no text and no images, but vector drawing       -> OCR (outlined text, signatures, forms)
6. a little text and nothing else                  -> text layer
7. nothing at all                                   -> blank, skipped (and not "unread")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Route(StrEnum):
    TEXT = "text-layer"
    OCR = "ocr"
    BLANK = "blank"


@dataclass(frozen=True)
class RoutingRules:
    min_text_chars: int = 30
    scan_cover: float = 0.6  # set above 1 to never treat a text page as a scan
    force_ocr: bool = False


@dataclass(frozen=True)
class PageFacts:
    number: int
    text: str
    image_cover: float = 0.0  # share of the page covered by its largest image
    image_count: int = 0
    has_vector_ink: bool = False


def route_page(facts: PageFacts, rules: RoutingRules) -> tuple[Route, str]:
    chars = len("".join(facts.text.split()))
    if rules.force_ocr:
        return Route.OCR, "OCR forced"
    if facts.image_count and facts.image_cover >= rules.scan_cover:
        detail = f" (ignoring its {chars}-character text layer)" if chars else ""
        return Route.OCR, (f"an image covers {facts.image_cover:.0%} of the page: treated as a "
                           f"scan{detail}")
    if chars >= rules.min_text_chars:
        return Route.TEXT, f"text layer with {chars} characters"
    if facts.image_count:
        return Route.OCR, f"only {chars} characters of text, plus {facts.image_count} image(s)"
    if facts.has_vector_ink:
        return Route.OCR, "no text layer, but the page has vector drawing"
    if chars:
        return Route.TEXT, f"short text layer ({chars} characters) and nothing else on the page"
    return Route.BLANK, "no text, images or drawing"
