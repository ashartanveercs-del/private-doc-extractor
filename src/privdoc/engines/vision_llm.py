"""Read a page image with a vision LLM: transcription, legibility and fields in one call."""

from __future__ import annotations

import base64
import time

from PIL import Image

from .. import prompts
from ..document import downscale, encode_png
from ..errors import EngineError
from ..llm import ChatClient
from ..pricing import Prices
from ..records import PageReading, Usage
from ..schema import LEGIBILITY, DocSchema


class VisionLLMReader:
    extracts_fields = True
    render_dpi = 200

    def __init__(self, client: ChatClient, model: str, *, prices: Prices, max_px: int = 2000,
                 detail: str = "high") -> None:
        self.client = client
        self.model = model
        self.prices = prices
        self.max_px = max_px
        self.detail = detail
        self.destination = client.destination
        self.id = f"vision-llm:{model}"
        self.fingerprint = f"vision-llm:{self.destination.host}:{model}:{max_px}:{detail}"

    def read(self, image: Image.Image, page: int, pages_total: int,
             schema: DocSchema) -> PageReading:
        started = time.perf_counter()
        png = encode_png(downscale(image, self.max_px))
        user = [
            {"type": "text", "text": prompts.vision_user(page, pages_total, schema)},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                "detail": self.detail}},
        ]
        result = self.client.complete_json(model=self.model, system=prompts.VISION_SYSTEM,
                                           user=user, schema_name="page_reading",
                                           schema=schema.page_schema())
        text, fields = result.data.get("transcription"), result.data.get("fields")
        if not isinstance(text, str) or not isinstance(fields, dict):
            raise EngineError(f"{self.model} returned an incomplete page reading")
        legibility = result.data.get("legibility")
        cost = self.prices.llm_cost(self.model, result.input_tokens, result.output_tokens)
        return PageReading(
            page=page, reader=self.id, text=text,
            legibility=legibility if legibility in LEGIBILITY else "fair", fields=fields,
            usage=Usage(llm_calls=1, input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens, cost_usd=cost),
            seconds=round(time.perf_counter() - started, 3))
