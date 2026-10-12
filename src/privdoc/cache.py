"""Content-addressed result cache.

Keys are SHA-256 hashes over exactly what determines a result: the document's content hash,
the page, the engine and its settings, the schema's fingerprint and the prompt version. Change
any of them (a new prompt, a stricter schema, a different model) and the old entry is simply
never looked up again; nothing needs invalidating by hand.

What is stored is the reading itself, which includes page text and field values in plaintext.
The directory is created private to the user (0700, files 0600), entries expire after a TTL,
only successful readings are written, and caching can be turned off entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 30 * 24 * 3600


def cache_key(**parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)  # also tightens a directory that already existed
    except OSError:
        pass


class Cache:
    def __init__(self, root: str | Path | None, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root) if root else None
        self.ttl = ttl_seconds
        self.clock = clock

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self._path(key)
        try:
            age = self.clock() - path.stat().st_mtime
        except OSError:
            return None
        if age > self.ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        return data if isinstance(data, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Best effort: a cache that cannot be written must not fail an extraction."""
        if self.root is None:
            return
        path = self._path(key)
        try:
            _private_dir(self.root)
            _private_dir(path.parent)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(value, fh, ensure_ascii=False)
                os.chmod(tmp, 0o600)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as err:
            log.warning("could not write cache entry: %s", type(err).__name__)

    def purge(self) -> int:
        if self.root is None or not self.root.exists():
            return 0
        removed = 0
        for path in self.root.glob("??/*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed
