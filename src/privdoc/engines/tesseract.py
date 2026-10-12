"""Local OCR with the Tesseract binary. Nothing leaves the machine.

The page is written to a private temporary directory (0700 on POSIX) only for as long as
Tesseract runs, then deleted. TSV output gives per-word confidences, which become the page's
legibility. The escalation rung (``clean=True``) re-reads at a higher resolution after
contrast stretching, denoising and Otsu binarisation, which helps faint or speckled scans.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from ..document import encode_png
from ..errors import EngineError
from ..records import PageReading, Usage
from ..schema import DocSchema
from .base import legibility_from


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


def run_process(cmd: list[str], timeout: float) -> ProcessResult:
    done = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    return ProcessResult(done.returncode, done.stdout.decode("utf-8", "replace"),
                         done.stderr.decode("utf-8", "replace"))


Runner = Callable[[list[str], float], ProcessResult]


def parse_tsv(tsv: str) -> tuple[str, float | None, int]:
    """Tesseract TSV -> (text in reading order, mean word confidence, word count)."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t", 11)
        if len(cols) < 12 or cols[0] != "5" or not cols[11].strip():
            continue
        try:
            key = (int(cols[2]), int(cols[3]), int(cols[4]))
            confidence = float(cols[10])
        except ValueError:
            continue
        lines.setdefault(key, []).append(cols[11].strip())
        if confidence >= 0:
            confidences.append(confidence)
    text = "\n".join(" ".join(words) for words in lines.values())
    mean = sum(confidences) / len(confidences) if confidences else None
    return text, mean, sum(len(words) for words in lines.values())


def _otsu(image: Image.Image) -> int:
    histogram = image.histogram()[:256]
    total = sum(histogram)
    weighted_total = sum(i * count for i, count in enumerate(histogram))
    best, best_threshold, background, weighted_background = -1.0, 127, 0, 0.0
    for threshold, count in enumerate(histogram):
        background += count
        if not background:
            continue
        foreground = total - background
        if not foreground:
            break
        weighted_background += threshold * count
        mean_bg = weighted_background / background
        mean_fg = (weighted_total - weighted_background) / foreground
        spread = background * foreground * (mean_bg - mean_fg) ** 2
        if spread > best:
            best, best_threshold = spread, threshold
    return best_threshold


def clean_for_ocr(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=1).filter(ImageFilter.MedianFilter(3))
    threshold = _otsu(gray)
    return gray.point(lambda value: 255 if value > threshold else 0)


class TesseractReader:
    extracts_fields = False
    destination = None

    def __init__(self, *, command: str = "tesseract", lang: str = "eng", psm: int = 3,
                 dpi: int = 300, clean: bool = False, runner: Runner | None = None,
                 timeout: float = 120.0) -> None:
        self.command = command
        self.lang = lang
        self.psm = psm
        self.render_dpi = dpi
        self.clean = clean
        self.runner = runner
        self.timeout = timeout
        self.id = f"tesseract:{lang}:psm{psm}:{dpi}dpi" + ("+clean" if clean else "")
        self.fingerprint = f"{self.id}:{command}"

    def read(self, image: Image.Image, page: int, pages_total: int,
             schema: DocSchema) -> PageReading:
        started = time.perf_counter()
        prepared = clean_for_ocr(image) if self.clean else image
        runner = self.runner if self.runner is not None else run_process
        workdir = Path(tempfile.mkdtemp(prefix="privdoc-"))
        try:
            source = workdir / f"page-{page:04d}.png"
            source.write_bytes(encode_png(prepared))
            cmd = [self.command, str(source), "stdout", "-l", self.lang, "--psm", str(self.psm),
                   "--dpi", str(self.render_dpi), "tsv"]
            try:
                result = runner(cmd, self.timeout)
            except FileNotFoundError:
                raise EngineError("tesseract is not installed (or not on PATH)") from None
            except subprocess.TimeoutExpired:
                raise EngineError(f"tesseract timed out on page {page}") from None
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if result.returncode != 0:
            first = (result.stderr.strip().splitlines() or ["unknown error"])[0][:200]
            raise EngineError(f"tesseract failed on page {page}: {first}")
        text, confidence, words = parse_tsv(result.stdout)
        legibility = legibility_from(confidence, good=80, fair=60) if words else "poor"
        return PageReading(page=page, reader=self.id, text=text, legibility=legibility,
                           usage=Usage(ocr_pages=1, cost_usd=0.0),
                           seconds=round(time.perf_counter() - started, 3))
