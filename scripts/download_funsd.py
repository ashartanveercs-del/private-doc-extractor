"""Download the FUNSD dataset for `privdoc bench --dataset funsd`.

FUNSD ("Form Understanding in Noisy Scanned Documents", Jaume et al., ICDAR-OST 2019) is a
public set of scanned forms with annotated, linked question/answer pairs. Read the dataset's own
license terms on its project page before using it.

This script only downloads the archive, through privdoc's SSRF-safe fetcher (size-capped, type-
checked), to a directory OUTSIDE this repository, and unpacks it with zip-slip and zip-bomb
checks. Nothing is ever vendored into the repository.

    python scripts/download_funsd.py --dest ~/.cache/privdoc/datasets
    privdoc bench --dataset funsd --dataset-dir ~/.cache/privdoc/datasets --tier cloud
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privdoc.datasets import FUNSD_URL, download_funsd
from privdoc.errors import PrivdocError
from privdoc.fetch import FetchPolicy, SafeFetcher

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None, *, fetcher=None) -> int:
    parser = argparse.ArgumentParser(description="Download FUNSD for the benchmark.")
    parser.add_argument("--dest", type=Path,
                        default=Path.home() / ".cache" / "privdoc" / "datasets",
                        help="where to put it (must be outside this repository)")
    parser.add_argument("--url", default=FUNSD_URL)
    parser.add_argument("--sha256", help="expected SHA-256 of the archive, if you have one")
    args = parser.parse_args(argv)
    fetcher = fetcher or SafeFetcher(FetchPolicy(max_bytes=100 * 1024 * 1024, timeout=300,
                                                 allowed_kinds=frozenset({"zip"})))
    try:
        root = download_funsd(args.dest.expanduser(), fetcher=fetcher, url=args.url,
                              sha256=args.sha256, repo_root=REPO_ROOT)
    except PrivdocError as err:
        print(f"download_funsd: error: {err}", file=sys.stderr)
        return err.exit_code
    print(f"FUNSD is ready at {root}")
    print(f"next: privdoc bench --dataset funsd --dataset-dir {root} --tier cloud")
    return 0


if __name__ == "__main__":
    sys.exit(main())
