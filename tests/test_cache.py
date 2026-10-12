import os
import stat
import time

import pytest

from privdoc.cache import Cache, cache_key


def test_keys_are_stable_hashes_that_ignore_argument_order():
    a = cache_key(kind="page", prompt="2026-10-01", reader="tesseract", doc="ab12", page=1)
    b = cache_key(page=1, doc="ab12", reader="tesseract", prompt="2026-10-01", kind="page")
    assert a == b and len(a) == 64 and int(a, 16) >= 0
    assert a != cache_key(kind="page", prompt="2026-10-02", reader="tesseract", doc="ab12", page=1)


def test_round_trip(tmp_path):
    cache = Cache(tmp_path / "cache")
    key = cache_key(doc="x")
    assert cache.get(key) is None
    cache.put(key, {"text": "Invoice #: INV-1042", "n": 1})
    assert cache.get(key) == {"text": "Invoice #: INV-1042", "n": 1}


def test_file_names_are_hashes_never_document_content(tmp_path):
    cache = Cache(tmp_path)
    cache.put(cache_key(doc="x"), {"text": "Cedar Lane Plumbing"})
    names = [p.name for p in tmp_path.rglob("*")]
    assert all("Cedar" not in n for n in names)


def test_disabled_cache_does_nothing(tmp_path):
    cache = Cache(None)
    cache.put("k" * 64, {"a": 1})
    assert cache.get("k" * 64) is None and cache.purge() == 0 and not cache.enabled


def test_entries_expire(tmp_path):
    now = [time.time()]
    cache = Cache(tmp_path, ttl_seconds=60, clock=lambda: now[0])
    key = cache_key(doc="x")
    cache.put(key, {"a": 1})
    now[0] += 61
    assert cache.get(key) is None
    assert not list(tmp_path.rglob("*.json"))


def test_corrupt_entries_are_misses_and_are_removed(tmp_path):
    cache = Cache(tmp_path)
    key = cache_key(doc="x")
    cache.put(key, {"a": 1})
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None and not path.exists()


def test_purge_removes_everything_and_counts(tmp_path):
    cache = Cache(tmp_path)
    for i in range(3):
        cache.put(cache_key(doc=str(i)), {"i": i})
    assert cache.purge() == 3
    assert not list(tmp_path.rglob("*.json"))


def test_writes_leave_no_temporary_files(tmp_path):
    cache = Cache(tmp_path)
    cache.put(cache_key(doc="x"), {"a": 1})
    assert [p.suffix for p in tmp_path.rglob("*") if p.is_file()] == [".json"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_cache_is_private_to_the_user(tmp_path):
    root = tmp_path / "cache"
    cache = Cache(root)
    cache.put(cache_key(doc="x"), {"a": 1})
    entry = next(root.rglob("*.json"))
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600
