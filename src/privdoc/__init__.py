"""private-doc-extractor: document-to-JSON extraction with privacy tiers.

    import privdoc
    result = privdoc.extract("invoice.pdf", schema="invoice")     # local tier unless configured
    result.data, result.valid, result.fields["total"].status
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import Settings
from .pipeline import ExtractionResult, ExtractOptions, Pipeline
from .privacy import Tier
from .schema import DocSchema, load_schema

__all__ = ["DocSchema", "ExtractOptions", "ExtractionResult", "Pipeline", "Settings", "Tier",
           "__version__", "extract", "load_schema"]


def extract(source: str | Path | bytes, schema: str | DocSchema, *,
            tier: Tier | str | None = None, settings: Settings | None = None,
            **options: Any) -> ExtractionResult:
    """Extract with settings read from the environment (the local tier unless configured)."""
    settings = settings or Settings.from_env(os.environ)
    return Pipeline(settings).extract(source, schema, ExtractOptions(tier=tier, **options))
