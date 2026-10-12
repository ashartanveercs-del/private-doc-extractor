"""Per-page routing: read the text layer for free, OCR only what is really an image.

The rules, in order:

1. ``force_ocr``                                   -> OCR
2. one image covers most of the page (>= 60 %)    -> OCR, unless most of the text layer is
   painted on top of that image. This catches "searchable" scans: a copier adds an OCR layer
   that holds the printed labels but not what was handwritten or stamped, so trusting it
   silently drops the values. That layer is invisible (text render mode 3) or hidden under the
   page image. Typed text drawn over a full-page background or letterhead image is visible, so
   such a page keeps its exact, free text layer.
3. a real text layer (>= 30 characters)            -> text layer (free, exact)
4. little or no text, but images                   -> OCR
5. no text and no images, but vector drawing       -> OCR (outlined text, signatures, forms)
6. a little text and nothing else                  -> text layer
7. nothing at all                                   -> blank, skipped (and not "unread")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# How much of a text layer must be painted on top of a page-sized image for the page to keep
# its text layer: all of it for typed text over a background, none of it for a copier's layer.
VISIBLE_SHARE = 0.5


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
    # share of the text layer's characters actually painted on top (not invisible, not under a
    # later image); None when it was not measured
    visible_text_share: float | None = None


def route_page(facts: PageFacts, rules: RoutingRules) -> tuple[Route, str]:
    chars = len("".join(facts.text.split()))
    if rules.force_ocr:
        return Route.OCR, "OCR forced"
    if facts.image_count and facts.image_cover >= rules.scan_cover:
        cover, shown = f"{facts.image_cover:.0%}", facts.visible_text_share
        if chars >= rules.min_text_chars and shown is not None and shown >= VISIBLE_SHARE:
            return Route.TEXT, (f"text layer with {chars} characters, painted over an image "
                                f"that covers {cover} of the page")
        if not chars:
            detail = ""
        elif shown is None:
            detail = f" (ignoring its {chars}-character text layer)"
        else:
            detail = (f" (its {chars}-character text layer is mostly invisible or under the "
                      "image)")
        return Route.OCR, f"an image covers {cover} of the page: treated as a scan{detail}"
    if chars >= rules.min_text_chars:
        return Route.TEXT, f"text layer with {chars} characters"
    if facts.image_count:
        return Route.OCR, f"only {chars} characters of text, plus {facts.image_count} image(s)"
    if facts.has_vector_ink:
        return Route.OCR, "no text layer, but the page has vector drawing"
    if chars:
        return Route.TEXT, f"short text layer ({chars} characters) and nothing else on the page"
    return Route.BLANK, "no text, images or drawing"
