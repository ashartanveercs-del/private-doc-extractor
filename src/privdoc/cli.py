"""Command line interface.

    privdoc extract invoice.pdf --schema invoice --tier local
    privdoc check                      # what each tier would use; sends nothing
    privdoc synth out/ --kind receipt  # synthetic documents with ground truth
    privdoc bench --tier local         # accuracy / hallucination / cost / speed
    privdoc cache purge

Exit codes: 0 extracted and schema-valid, 1 extracted but incomplete or invalid, 2 usage error
or refused by the privacy tier / SSRF guard, 3 the document could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._version import __version__
from .cache import Cache
from .config import Settings, read_env_file
from .errors import EgressBlocked, PrivdocError, UsageError
from .pipeline import ExtractOptions, Pipeline
from .plan import ENGINES, EXTRACTORS, build_plan
from .privacy import Tier
from .synth import KINDS, VARIANTS, make_doc, render


class _NoNetwork:
    def post_json(self, url, *, headers, body, timeout):
        raise EgressBlocked("`privdoc check` never sends anything")


def _emit(payload: Any, compact: bool = False) -> None:
    text = json.dumps(payload, indent=None if compact else 2, ensure_ascii=False)
    try:
        print(text)
    except UnicodeEncodeError:  # a legacy console code page
        print(json.dumps(payload, indent=None if compact else 2))


def _extract(args: argparse.Namespace, settings: Settings, kwargs: dict[str, Any]) -> int:
    options = ExtractOptions(tier=args.tier, engine=args.engine, extractor=args.extractor,
                             pages=args.pages, force_ocr=args.force_ocr,
                             escalate=not args.no_escalate, use_cache=not args.no_cache,
                             include_text=args.include_text)
    result = Pipeline(settings, **kwargs).extract(args.source, args.schema, options)
    if args.out:
        text = json.dumps(result.to_dict(), indent=None if args.compact else 2,
                          ensure_ascii=False)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        found = sum(f.status == "found" for f in result.fields.values())
        print(f"{found}/{len(result.fields)} fields found, valid={str(result.valid).lower()}, "
              f"complete={str(result.complete).lower()}; written to {args.out}",
              file=sys.stderr)
    else:
        _emit(result.to_dict(), args.compact)
    if not result.ok:
        return 3
    return 0 if result.valid else 1


def _check(args: argparse.Namespace, settings: Settings, kwargs: dict[str, Any]) -> int:
    which = kwargs.get("which", shutil.which)
    tiers = {}
    for tier in Tier:
        plan = build_plan(settings, tier, transport=_NoNetwork(), which=which,
                          tesseract_runner=kwargs.get("tesseract_runner"))
        tiers[str(tier)] = plan.describe()
    _emit({"version": __version__, "default_tier": str(settings.tier),
           "tesseract": which(settings.tesseract_cmd),
           "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
           "tiers": tiers})
    return 0


def _synth(args: argparse.Namespace, settings: Settings, kwargs: dict[str, Any]) -> int:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for kind in args.kind or list(KINDS):
        for seed in range(args.seed, args.seed + args.count):
            doc = make_doc(kind, seed)
            for variant in args.variant or ["text", "scan"]:
                rendered = render(doc, variant)
                (out / rendered.name).write_bytes(rendered.data)
                truth = {"file": rendered.name, "kind": kind, "seed": seed, "variant": variant,
                         "truth": doc.truth, "absent": doc.absent}
                stem = rendered.name.rsplit(".", 1)[0]
                (out / f"{stem}.truth.json").write_text(json.dumps(truth, indent=2) + "\n",
                                                        encoding="utf-8")
                written += 1
    print(f"wrote {written} synthetic documents (and their ground truth) to {out}")
    return 0


def _bench(args: argparse.Namespace, settings: Settings, kwargs: dict[str, Any]) -> int:
    from .bench import run_cli

    return run_cli(args, settings, kwargs)


def _cache(args: argparse.Namespace, settings: Settings, kwargs: dict[str, Any]) -> int:
    if args.action == "path":
        print(settings.cache_dir if settings.cache_dir else "caching is off")
        return 0
    removed = Cache(settings.cache_dir).purge()
    print(f"removed {removed} cached result(s)"
          + (f" from {settings.cache_dir}" if settings.cache_dir else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privdoc", description="Document-to-JSON extraction with privacy tiers.")
    parser.add_argument("--version", action="version", version=f"privdoc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--env-file", help="read settings from this .env file")
        p.set_defaults(handler=handler)
        return p

    extract = command("extract", _extract, "Extract schema-valid JSON from a PDF or image.")
    extract.add_argument("source", help="a file path, or an http(s) link (not in the local tier)")
    extract.add_argument("--schema", required=True,
                         help="built-in name (invoice, receipt, tax_form) or a JSON schema file")
    extract.add_argument("--tier", help="cloud, own-cloud or local (default: PRIVDOC_TIER, "
                                        "else local)")
    extract.add_argument("--engine", choices=ENGINES, default="auto",
                         help="OCR / vision engine for pages without a text layer")
    extract.add_argument("--extractor", choices=EXTRACTORS, default="auto",
                         help="how fields are pulled from text")
    extract.add_argument("--pages", help="pages to read, e.g. 1-3,5 or 4-")
    extract.add_argument("--force-ocr", action="store_true",
                         help="OCR every page, even ones with a text layer")
    extract.add_argument("--no-escalate", action="store_true",
                         help="never re-read doubtful pages with a stronger engine")
    extract.add_argument("--no-cache", action="store_true", help="don't read or write the cache")
    extract.add_argument("--include-text", action="store_true",
                         help="include each page's text in the output")
    extract.add_argument("--out", help="write the JSON result to this file")
    extract.add_argument("--compact", action="store_true", help="one-line JSON")

    command("check", _check, "Show which engines each tier would use. Sends nothing.")

    synth = command("synth", _synth, "Write synthetic documents with known ground truth.")
    synth.add_argument("outdir")
    synth.add_argument("--kind", action="append", choices=KINDS)
    synth.add_argument("--variant", action="append", choices=VARIANTS)
    synth.add_argument("--count", type=int, default=3)
    synth.add_argument("--seed", type=int, default=0)

    bench = command("bench", _bench, "Measure field accuracy, hallucination, cost and speed.")
    bench.add_argument("--tier", help="cloud, own-cloud or local")
    bench.add_argument("--engine", choices=ENGINES, default="auto")
    bench.add_argument("--extractor", choices=EXTRACTORS, default="auto")
    bench.add_argument("--kinds", default=",".join(KINDS),
                       help="comma-separated: " + ", ".join(KINDS))
    bench.add_argument("--variants", default="text,scan,searchable-scan,mixed,photo",
                       help="comma-separated: " + ", ".join(VARIANTS))
    bench.add_argument("--per-kind", type=int, default=4, help="documents per kind and variant")
    bench.add_argument("--seed", type=int, default=1000)
    bench.add_argument("--no-escalate", action="store_true")
    bench.add_argument("--dataset", choices=("synthetic", "funsd"), default="synthetic")
    bench.add_argument("--dataset-dir", help="where scripts/download_funsd.py put FUNSD")
    bench.add_argument("--limit", type=int, default=10, help="FUNSD forms to use")
    bench.add_argument("--json", dest="json_out", help="also write the full report here")

    cache = command("cache", _cache, "Manage the local result cache.")
    cache.add_argument("action", choices=("purge", "path"))
    return parser


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None,
         pipeline_kwargs: dict[str, Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environ = dict(os.environ if env is None else env)
    try:
        if args.env_file:
            try:
                file_values = read_env_file(args.env_file)
            except OSError:
                raise UsageError(f"cannot read env file {args.env_file}") from None
            environ = {**file_values, **environ}  # the real environment wins
        settings = Settings.from_env(environ)
        return args.handler(args, settings, dict(pipeline_kwargs or {}))
    except PrivdocError as err:
        print(f"privdoc: error: {err}", file=sys.stderr)
        return err.exit_code
    except Exception as err:
        if environ.get("PRIVDOC_DEBUG") == "1":
            raise
        print(f"privdoc: error: unexpected {type(err).__name__}; set PRIVDOC_DEBUG=1 for "
              "details", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
