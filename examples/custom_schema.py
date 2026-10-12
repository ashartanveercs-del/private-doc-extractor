"""Use your own JSON schema: a printed support ticket, extracted in the local tier.

    python examples/custom_schema.py [--out DIR]

The schema (examples/support_ticket.schema.json) uses `pattern`, `format`, `enum` and
`x-labels`. Values are coerced ("03/02/2026" -> "2026-03-02", "[x]" -> true, "$149.00" ->
149.0) and then validated; the printed result is schema-valid JSON plus per-field status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from privdoc import ExtractOptions, Pipeline, Settings, Tier, load_schema
from privdoc.synth import PdfPage, TextRun, write_pdf

HERE = Path(__file__).resolve().parent
TICKET = [
    "Summit Bike Repair - service desk",
    "Ticket: TCK-04217",
    "Opened: 03/02/2026",
    "Customer: Casey Nguyen",
    "Email: casey@example.com",
    "Priority: High",
    "Device: Commuter bike, 8-speed",
    "Under warranty: [x]",
    "Estimate: $149.00",
    "Notes: rear derailleur skips under load; customer will pick up after 5 pm.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    path = args.out / "support-ticket.pdf"
    runs = [TextRun(72, 90 + 20 * i, text, size=11) for i, text in enumerate(TICKET)]
    path.write_bytes(write_pdf([PdfPage(runs=runs)]))

    schema = load_schema(str(HERE / "support_ticket.schema.json"))
    result = Pipeline(Settings()).extract(path, schema, ExtractOptions(tier=Tier.LOCAL))
    print(json.dumps({"valid": result.valid, "data": result.data,
                      "statuses": {n: f.status for n, f in result.fields.items()}}, indent=2))


if __name__ == "__main__":
    main()
