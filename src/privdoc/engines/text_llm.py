"""Extract fields from page text with an LLM (and repair answers that failed validation)."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from .. import prompts
from ..errors import EngineError
from ..llm import ChatClient, ChatResult
from ..pricing import Prices
from ..records import FieldOutput, Usage
from ..schema import DocSchema
from .base import PageText


class TextLLMExtractor:
    can_repair = True

    def __init__(self, client: ChatClient, model: str, *, prices: Prices,
                 max_chars: int = 120_000) -> None:
        self.client = client
        self.model = model
        self.prices = prices
        self.max_chars = max_chars
        self.destination = client.destination
        self.id = f"text-llm:{model}"
        self.fingerprint = f"text-llm:{self.destination.host}:{model}:{max_chars}"

    def _run(self, system: str, user: str, name: str, schema: DocSchema,
             truncated: bool) -> FieldOutput:
        started = time.perf_counter()
        result: ChatResult = self.client.complete_json(
            model=self.model, system=system, user=user, schema_name=name,
            schema=schema.document_schema())
        fields = result.data.get("fields")
        if not isinstance(fields, dict):
            raise EngineError(f"{self.model} returned no fields")
        cost = self.prices.llm_cost(self.model, result.input_tokens, result.output_tokens)
        usage = Usage(llm_calls=1, input_tokens=result.input_tokens,
                      output_tokens=result.output_tokens, cost_usd=cost)
        return FieldOutput(fields=fields, usage=usage, truncated=truncated,
                           seconds=round(time.perf_counter() - started, 3))

    def extract(self, pages: Sequence[PageText], schema: DocSchema) -> FieldOutput:
        block, truncated = prompts.document_block([(p.page, p.text) for p in pages],
                                                  self.max_chars)
        return self._run(prompts.TEXT_SYSTEM, prompts.text_user(schema, block),
                         "document_fields", schema, truncated)

    def repair(self, pages: Sequence[PageText], schema: DocSchema, previous: Mapping[str, Any],
               problems: Sequence[str]) -> FieldOutput:
        block, truncated = prompts.document_block([(p.page, p.text) for p in pages],
                                                  self.max_chars)
        return self._run(prompts.REPAIR_SYSTEM,
                         prompts.repair_user(schema, block, previous, problems),
                         "field_repair", schema, truncated)
