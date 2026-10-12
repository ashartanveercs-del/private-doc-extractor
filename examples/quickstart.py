"""Extract an invoice in the local tier: no API keys, nothing leaves this machine.

    python examples/quickstart.py [--out DIR]

It generates two synthetic invoices (one typed PDF, one scanned PDF), then extracts both. The
typed one is read from its text layer for free. The scanned one needs an OCR engine: with
Tesseract installed it is OCR'd locally; without it, the result says exactly which pages could
not be read instead of pretending the fields are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privdoc import ExtractOptions, Pipeline, Settings, Tier
from privdoc.synth import make_doc, render


def show(title: str, result) -> None:
    print(f"\n== {title}: valid={result.valid} complete={result.complete} "
          f"cost=${result.usage.cost_usd or 0:.4f} sent_to={result.data_sent_to or 'nowhere'}")
    for page in result.pages:
        print(f"   page {page.page}: {page.route} ({page.reason}) -> {page.status}")
    for name in ("vendor_name", "invoice_number", "invoice_date", "total"):
        field = result.fields[name]
        print(f"   {name:<15} {field.status:<12} {field.value!r}")
    for unread in result.pages_unread:
        print(f"   unread page {unread['page']}: {unread['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    doc = make_doc("invoice", seed=42)
    pipeline = Pipeline(Settings())  # defaults: local tier, cache off
    for variant in ("text", "scan"):
        rendered = render(doc, variant)
        path = args.out / rendered.name
        path.write_bytes(rendered.data)
        result = pipeline.extract(path, "invoice", ExtractOptions(tier=Tier.LOCAL))
        show(path.name, result)
    print(f"\nground truth: invoice_number={doc.truth['invoice_number']!r} "
          f"total={doc.truth['total']!r}")


if __name__ == "__main__":
    main()
