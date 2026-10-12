"""Choose the engines for a run, within what the privacy tier allows.

The plan is two ladders. ``readers`` turn page images into text (and sometimes fields): the
first rung reads every scanned page, later rungs re-read only the pages that failed or came
back doubtful. ``extractors`` turn text into fields, with the same idea. By tier:

  cloud      vision LLM -> larger vision LLM          text LLM -> larger text LLM
             (else Cloud Vision, else Tesseract)      (else the rule extractor)
  own-cloud  Cloud Vision in your project -> an LLM only if you host it yourself
             (else Tesseract)                         rules (or your own LLM)
  local      Tesseract -> Tesseract at 400 dpi on a cleaned-up image      rules

Explicitly asking for an engine the tier does not allow is refused here, before any document is
opened; anything an engine would send is registered with the EgressGate, which blocks the rest.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .engines.base import FieldExtractor, PageReader
from .engines.cloud_vision import CloudVisionReader
from .engines.rules import RuleExtractor
from .engines.tesseract import Runner, TesseractReader
from .engines.text_llm import TextLLMExtractor
from .engines.vision_llm import VisionLLMReader
from .errors import PrivacyViolation, UsageError
from .llm import ChatClient
from .privacy import Destination, EgressGate, Tier, permits
from .transport import Transport

ENGINES = ("auto", "vision-llm", "cloud-ocr", "tesseract")
EXTRACTORS = ("auto", "llm", "rules")
_AUTO_ORDER = {
    Tier.CLOUD: ("vision-llm", "cloud-ocr", "tesseract"),
    Tier.OWN_CLOUD: ("cloud-ocr", "vision-llm", "tesseract"),
    Tier.LOCAL: ("tesseract",),
}
_NO_OCR_HINT = {
    Tier.LOCAL: "install Tesseract (or set PRIVDOC_TESSERACT_CMD)",
    Tier.OWN_CLOUD: "configure Cloud Vision (PRIVDOC_GCV_API_KEY) or install Tesseract",
    Tier.CLOUD: "configure an LLM (PRIVDOC_LLM_MODEL) or Cloud Vision, or install Tesseract",
}


@dataclass
class Plan:
    tier: Tier
    gate: EgressGate
    readers: list[PageReader]
    extractors: list[FieldExtractor]
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {"tier": str(self.tier), "summary": self.tier.summary,
                "ocr": [r.id for r in self.readers],
                "extract": [e.id for e in self.extractors],
                "destinations": self.gate.destinations, "notes": self.notes}


def _require(name: str, tier: Tier, destination: Destination, configured: bool,
             hint: str) -> None:
    if not permits(tier, destination):
        raise PrivacyViolation(f"{name} would send pages to {destination.host}, a "
                               f"{destination.trust} service; the {tier} tier {tier.summary}")
    if not configured:
        raise UsageError(f"{name} is not configured: {hint}")


def build_plan(settings: Settings, tier: Tier | None = None, *, engine: str = "auto",
               extractor: str = "auto", transport: Transport | None = None,
               tesseract_runner: Runner | None = None,
               which: Callable[[str], str | None] | None = None,
               sleep: Callable[[float], None] = time.sleep) -> Plan:
    tier = tier or settings.tier
    which = which or shutil.which  # looked up per call, so it can be swapped in tests
    if engine not in ENGINES:
        raise UsageError(f"unknown engine {engine!r}; choose one of {', '.join(ENGINES)}")
    if extractor not in EXTRACTORS:
        raise UsageError(f"unknown extractor {extractor!r}; choose one of "
                         + ", ".join(EXTRACTORS))
    gate = EgressGate(tier, transport)
    notes: list[str] = []

    client = ChatClient(gate, settings.llm_base_url, settings.llm_api_key,
                        trust=settings.llm_trust, temperature=settings.llm_temperature,
                        sleep=sleep)
    llm_configured = bool(settings.llm_model)
    llm_usable = llm_configured and permits(tier, client.destination)
    models = [m for m in dict.fromkeys([settings.llm_model, settings.llm_escalation_model]) if m]
    strongest = settings.llm_escalation_model or settings.llm_model

    gcv = CloudVisionReader(gate, endpoint=settings.gcv_endpoint, api_key=settings.gcv_api_key,
                            access_token=settings.gcv_access_token,
                            quota_project=settings.gcv_quota_project, lang=settings.ocr_lang,
                            prices=settings.prices)
    gcv_configured = bool(settings.gcv_api_key or settings.gcv_access_token)
    tesseract_found = which(settings.tesseract_cmd) is not None

    def vision(model: str) -> VisionLLMReader:
        return VisionLLMReader(client, model, prices=settings.prices, max_px=settings.llm_max_px)

    if engine == "auto":
        available = {"vision-llm": llm_usable, "tesseract": tesseract_found,
                     "cloud-ocr": gcv_configured and permits(tier, gcv.destination)}
        chosen = next((name for name in _AUTO_ORDER[tier] if available[name]), None)
    else:
        if engine == "vision-llm":
            _require("the vision-llm engine", tier, client.destination, llm_configured,
                     "set PRIVDOC_LLM_MODEL")
        elif engine == "cloud-ocr":
            _require("the cloud-ocr engine", tier, gcv.destination, gcv_configured,
                     "set PRIVDOC_GCV_API_KEY or PRIVDOC_GCV_ACCESS_TOKEN")
        elif not tesseract_found:
            raise UsageError(f"tesseract was not found ({settings.tesseract_cmd!r}); install "
                             "it or set PRIVDOC_TESSERACT_CMD")
        chosen = engine

    escalation = [vision(strongest)] if llm_usable else []
    readers: list[PageReader]
    if chosen == "vision-llm":
        readers = [vision(m) for m in models]
    elif chosen == "cloud-ocr":
        readers = [gcv, *escalation]
    elif chosen == "tesseract":
        tess = dict(command=settings.tesseract_cmd, lang=settings.tesseract_lang,
                    runner=tesseract_runner)
        readers = [TesseractReader(**tess), TesseractReader(**tess, psm=6, dpi=400, clean=True),
                   *escalation]
    else:
        readers = []
        notes.append(f"no OCR engine is available in the {tier} tier: {_NO_OCR_HINT[tier]}. "
                     "Pages without a text layer will be listed as unread.")

    rules = RuleExtractor(date_order=settings.date_order)
    extractors: list[FieldExtractor]
    if extractor == "llm":
        _require("the llm extractor", tier, client.destination, llm_configured,
                 "set PRIVDOC_LLM_MODEL")
    if extractor == "rules" or not llm_usable:
        extractors = [rules]
    else:
        extractors = [TextLLMExtractor(client, m, prices=settings.prices) for m in models]

    if llm_configured and not permits(tier, client.destination):
        notes.append(f"the LLM at {client.destination.host} is a {client.destination.trust} "
                     f"service, which the {tier} tier does not use")
    if gcv_configured and not permits(tier, gcv.destination):
        notes.append(f"Cloud Vision is configured, but the {tier} tier does not use it")
    for component in [*readers, *extractors]:
        if component.destination is not None:
            gate.allow(component.destination)
    return Plan(tier, gate, readers, extractors, notes)
