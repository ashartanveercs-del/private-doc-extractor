"""In-memory stand-ins for every external service: HTTP APIs, the LLM, Cloud OCR and Tesseract."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from privdoc.transport import HttpResponse


@dataclass
class SentRequest:
    url: str
    headers: dict[str, str]
    body: Any


class FakeTransport:
    """Records every request that made it past the EgressGate; `handler` scripts the reply."""

    def __init__(self, handler: Callable[[str, dict[str, str], Any], Any] | None = None) -> None:
        self.handler = handler or (lambda url, headers, body: (200, {}))
        self.requests: list[SentRequest] = []

    def post_json(self, url, *, headers, body, timeout):
        self.requests.append(SentRequest(url, dict(headers), body))
        out = self.handler(url, dict(headers), body)
        if isinstance(out, HttpResponse):
            return out
        if isinstance(out, Exception):
            raise out
        status, payload = out
        return HttpResponse(status=status, headers={}, body=json.dumps(payload).encode())


def fval(value=None, *, confidence="high", evidence=None, unreadable=False, page=None):
    """One field as an extraction model would report it."""
    if evidence is None and value is not None:
        evidence = str(value)
    out = {"value": value, "unreadable": unreadable,
           "confidence": "none" if value is None and not unreadable else confidence,
           "evidence": evidence}
    if page is not None:
        out["page"] = page
    return out


def chat_reply(content: dict | str, *, model="model-a", prompt_tokens=1000, completion_tokens=200,
               finish_reason="stop", refusal=None) -> dict:
    message = {"role": "assistant",
               "content": content if isinstance(content, str) else json.dumps(content)}
    if refusal is not None:
        message = {"role": "assistant", "content": None, "refusal": refusal}
    return {"id": "chatcmpl-test", "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


@dataclass
class LLMCall:
    model: str
    kind: str  # "page_reading" | "document_fields" | "field_repair"
    page: int | None
    pages: list[int]
    body: dict
    text: str


class FakeLLM:
    """An OpenAI-compatible chat endpoint.

    Answers are scripted per request kind:
      * page(model, page) -> {"transcription", "legibility", "fields"} for vision page reads
      * fields(model, pages, text) -> {"fields": {...}} for text extraction
      * repair(model, text) -> {"fields": {...}} for repair rounds
    A scripted answer may also be an Exception (raised as HTTP 500) or an int (HTTP status).
    """

    def __init__(self, *, page=None, fields=None, repair=None) -> None:
        self.page = page
        self.fields = fields
        self.repair = repair
        self.calls: list[LLMCall] = []

    def __call__(self, url: str, headers: dict[str, str], body: dict):
        kind = body["response_format"]["json_schema"]["name"]
        model = body["model"]
        user = body["messages"][-1]["content"]
        parts = user if isinstance(user, list) else [{"type": "text", "text": user}]
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        m = re.search(r"This is page (\d+) of", text)
        page = int(m.group(1)) if m else None
        pages = [int(n) for n in re.findall(r"=== page (\d+) ===", text)]
        self.calls.append(LLMCall(model, kind, page, pages, body, text))
        if kind == "page_reading":
            answer = self.page(model, page) if self.page else None
        elif kind == "document_fields":
            answer = self.fields(model, pages, text) if self.fields else None
        else:
            answer = self.repair(model, text) if self.repair else None
        if answer is None:
            return 500, {"error": {"message": f"no scripted answer for {kind}"}}
        if isinstance(answer, int):
            return answer, {"error": {"message": "scripted failure"}}
        if isinstance(answer, Exception):
            return 500, {"error": {"message": str(answer)}}
        return 200, chat_reply(answer, model=model)

    def of_kind(self, kind: str) -> list[LLMCall]:
        return [c for c in self.calls if c.kind == kind]


def tsv_for(text: str, conf: float = 95.0) -> str:
    """Tesseract's TSV output for `text`, one TSV word row per word."""
    rows = ["level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight"
            "\tconf\ttext", "1\t1\t0\t0\t0\t0\t0\t0\t1700\t2200\t-1\t"]
    for li, line in enumerate(text.splitlines(), start=1):
        rows.append(f"4\t1\t1\t1\t{li}\t0\t10\t{li * 40}\t500\t30\t-1\t")
        for wi, word in enumerate(line.split(), start=1):
            rows.append(f"5\t1\t1\t1\t{li}\t{wi}\t{wi * 60}\t{li * 40}\t50\t30\t{conf}\t{word}")
    return "\n".join(rows) + "\n"


@dataclass
class FakeTesseract:
    """Replaces the tesseract subprocess. Finds the page number in the input file name."""

    pages: dict[int, str] = field(default_factory=dict)
    conf: float | dict[int, float] = 95.0
    returncode: int = 0
    calls: list[list[str]] = field(default_factory=list)
    inputs: list[Path] = field(default_factory=list)

    def __call__(self, cmd: list[str], timeout: float):
        from privdoc.engines.tesseract import ProcessResult

        self.calls.append(list(cmd))
        path = Path(cmd[1])
        self.inputs.append(path)
        assert path.exists(), "tesseract input file must exist while it runs"
        m = re.search(r"page-(\d+)", path.name)
        page = int(m.group(1)) if m else 1
        if self.returncode:
            return ProcessResult(self.returncode, "", "Error: simulated failure\nmore detail")
        conf = self.conf.get(page, 95.0) if isinstance(self.conf, dict) else self.conf
        return ProcessResult(0, tsv_for(self.pages.get(page, ""), conf), "")
