"""Optional public benchmark data: the FUNSD form-understanding dataset.

FUNSD (Jaume et al., ICDAR-OST 2019) is a set of noisy scanned forms whose question/answer
pairs are annotated and linked. Each form becomes one benchmark case: every linked question is
a field (its printed label), its linked answers are the ground truth, and a few answered
questions borrowed from *other* forms are added as decoys, so asking for something that is not
on the page measures hallucination.

Nothing here is vendored: ``scripts/download_funsd.py`` downloads the archive to a directory
outside the repository, through the SSRF-safe fetcher, and unpacks it with zip-slip and
zip-bomb checks.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import DatasetError
from .schema import DocSchema

FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
_SLUG = re.compile(r"[^a-z0-9]+")


def safe_extract_zip(archive: bytes | Path, dest: Path, *, max_members: int = 20_000,
                     max_total_bytes: int = 1024 * 1024 * 1024,
                     max_ratio: float = 200.0) -> list[Path]:
    """Unpack without letting the archive write outside `dest` or expand without bound."""
    source = io.BytesIO(archive) if isinstance(archive, bytes) else Path(archive)
    root = dest.resolve()
    written: list[Path] = []
    try:
        zf = zipfile.ZipFile(source)
    except zipfile.BadZipFile:
        raise DatasetError("the download is not a valid zip archive") from None
    with zf:
        members = zf.infolist()
        if len(members) > max_members:
            raise DatasetError(f"the archive has more than {max_members} members")
        if sum(m.file_size for m in members) > max_total_bytes:
            raise DatasetError("the archive would unpack to more than "
                               f"{max_total_bytes / 1048576:g} MB, larger than allowed")
        for member in members:
            name = member.filename
            parts = PurePosixPath(name.replace("\\", "/")).parts
            if (not parts or name.startswith(("/", "\\")) or ":" in name
                    or any(part == ".." for part in parts)):
                raise DatasetError(f"unsafe path in archive: {name!r}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise DatasetError(f"the archive contains a symbolic link: {name!r}")
            if member.file_size > 1024 * 1024 and member.compress_size and (
                    member.file_size / member.compress_size > max_ratio):
                raise DatasetError(f"{name!r} has a suspicious compression ratio (zip bomb?)")
            target = root.joinpath(*parts).resolve()
            if not target.is_relative_to(root):
                raise DatasetError(f"unsafe path in archive: {name!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                copied = 0
                while chunk := src.read(65536):
                    copied += len(chunk)
                    if copied > member.file_size:
                        raise DatasetError(f"{name!r} is larger than its header claims")
                    out.write(chunk)
            written.append(target)
    return written


def find_funsd_root(path: Path) -> Path:
    for candidate in (path, path / "dataset", path / "funsd" / "dataset"):
        if (candidate / "testing_data" / "annotations").is_dir():
            return candidate
    raise DatasetError(f"no FUNSD layout under {path}: expected testing_data/annotations "
                       "(run scripts/download_funsd.py first)")


def download_funsd(dest: Path, *, fetcher: Any, url: str = FUNSD_URL, sha256: str | None = None,
                   repo_root: Path | None = None) -> Path:
    dest = dest.resolve()
    if repo_root is not None and dest.is_relative_to(repo_root.resolve()):
        raise DatasetError("refusing to download a dataset into the repository; choose a "
                           "directory outside it")
    fetched = fetcher.fetch(url)
    digest = hashlib.sha256(fetched.data).hexdigest()
    if sha256 and digest != sha256.lower():
        raise DatasetError(f"sha256 mismatch: expected {sha256}, got {digest}")
    target = dest / "funsd"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    safe_extract_zip(fetched.data, target)
    return find_funsd_root(target)


def _slug(question: str, taken: set[str]) -> str | None:
    base = _SLUG.sub("_", question.lower()).strip("_")[:40].strip("_")
    if not base:
        return None
    name, i = f"q_{base}", 2
    while name in taken:
        name, i = f"q_{base}_{i}", i + 1
    return name


def _pairs(entities: list[dict[str, Any]], max_fields: int) -> dict[str, tuple[str, str]]:
    by_id = {e.get("id"): e for e in entities}
    fields: dict[str, tuple[str, str]] = {}
    for entity in sorted(entities, key=lambda e: e.get("id", 0)):
        if entity.get("label") != "question":
            continue
        answer_ids = sorted(b for a, b in entity.get("linking") or []
                            if a == entity.get("id") and b in by_id
                            and by_id[b].get("label") == "answer")
        value = " ".join(by_id[b].get("text", "").strip() for b in answer_ids).strip()
        question = str(entity.get("text") or "").strip().rstrip(":").strip()[:80]
        name = _slug(question, set(fields)) if value and question else None
        if name:
            fields[name] = (question, value)
        if len(fields) >= max_fields:
            break
    return fields


def load_funsd(root: Path, *, split: str = "testing_data", limit: int | None = None,
               max_fields: int = 8, decoys: int = 2, seed: int = 0) -> list[Any]:
    from .bench import BenchCase
    from .grounding import normalize_text

    annotations = sorted((root / split / "annotations").glob("*.json"))[:limit]
    forms = []
    for path in annotations:
        entities = json.loads(path.read_text(encoding="utf-8")).get("form") or []
        image = root / split / "images" / f"{path.stem}.png"
        if not image.is_file():
            continue
        text = "\n".join(str(e.get("text") or "") for e in entities)
        forms.append((path.stem, image, text, _pairs(entities, max_fields)))
    rng = random.Random(seed)
    cases = []
    for stem, image, text, fields in forms:
        page = normalize_text(text)
        pool = sorted({(name, q) for other, _, _, f in forms if other != stem
                       for name, (q, _) in f.items()
                       if name not in fields and normalize_text(q) not in page})
        chosen = rng.sample(pool, min(decoys, len(pool)))
        properties = {name: {"type": "string", "description": f'The answer to "{q}"',
                             "x-labels": [q]}
                      for name, (q, _) in [*fields.items(), *[(n, (q, "")) for n, q in chosen]]}
        if not properties:
            continue
        cases.append(BenchCase(
            id=f"funsd-{stem}", kind="funsd", variant="scan", data=image.read_bytes(),
            schema=DocSchema({"type": "object", "properties": properties}),
            truth={name: value for name, (_, value) in fields.items()},
            source_text=text, pages=1))
    return cases
