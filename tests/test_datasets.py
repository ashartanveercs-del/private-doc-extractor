import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from privdoc.datasets import (
    FUNSD_URL,
    download_funsd,
    find_funsd_root,
    load_funsd,
    safe_extract_zip,
)
from privdoc.errors import DatasetError
from privdoc.fetch import Fetched


def zip_bytes(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (0o120777 << 16)
            z.writestr(info, "/etc/passwd")
    return buf.getvalue()


def test_a_normal_archive_is_extracted(tmp_path):
    written = safe_extract_zip(zip_bytes({"dataset/a.txt": b"hello", "dataset/sub/b.txt": b"x"}),
                               tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == [
        "dataset/a.txt", "dataset/sub/b.txt"]
    assert (tmp_path / "dataset" / "a.txt").read_bytes() == b"hello"


@pytest.mark.parametrize("members, symlink, message", [
    ({"../evil.txt": b"x"}, None, "unsafe path"),
    ({"/etc/evil.txt": b"x"}, None, "unsafe path"),
    ({"C:/evil.txt": b"x"}, None, "unsafe path"),
    ({"dataset\\..\\..\\evil.txt": b"x"}, None, "unsafe path"),
    ({"ok.txt": b"x"}, "dataset/link", "symbolic link"),
])
def test_zip_slip_and_links_are_refused(tmp_path, members, symlink, message):
    with pytest.raises(DatasetError, match=message):
        safe_extract_zip(zip_bytes(members, symlink), tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_archive_size_and_compression_ratio_limits(tmp_path):
    bomb = zip_bytes({"zeros.bin": b"\0" * 5_000_000})
    with pytest.raises(DatasetError, match="ratio"):
        safe_extract_zip(bomb, tmp_path, max_ratio=100)
    with pytest.raises(DatasetError, match="larger"):
        safe_extract_zip(zip_bytes({"a.bin": b"1" * 2000}), tmp_path, max_total_bytes=1000)
    with pytest.raises(DatasetError, match="members"):
        safe_extract_zip(zip_bytes({f"f{i}": b"" for i in range(5)}), tmp_path, max_members=3)


def write_funsd(root: Path) -> Path:
    """Two invented forms in FUNSD's annotation layout (not real data)."""
    split = root / "dataset" / "testing_data"
    (split / "annotations").mkdir(parents=True)
    (split / "images").mkdir(parents=True)
    forms = {
        "form_a": [
            {"id": 0, "label": "header", "text": "FAX COVER SHEET", "linking": []},
            {"id": 1, "label": "question", "text": "Date:", "linking": [[1, 2]]},
            {"id": 2, "label": "answer", "text": "03/14/1999", "linking": [[1, 2]]},
            {"id": 3, "label": "question", "text": "To:", "linking": [[3, 4], [3, 5]]},
            {"id": 4, "label": "answer", "text": "Jordan", "linking": [[3, 4]]},
            {"id": 5, "label": "answer", "text": "Lee", "linking": [[3, 5]]},
            {"id": 6, "label": "question", "text": "Pages:", "linking": []},
        ],
        "form_b": [
            {"id": 0, "label": "question", "text": "Fax Number:", "linking": [[0, 1]]},
            {"id": 1, "label": "answer", "text": "+1 555 000 0199", "linking": [[0, 1]]},
        ],
    }
    for stem, entities in forms.items():
        (split / "annotations" / f"{stem}.json").write_text(json.dumps({"form": entities}),
                                                             encoding="utf-8")
        Image.new("L", (100, 120), 255).save(split / "images" / f"{stem}.png")
    return root


def test_funsd_forms_become_cases_from_linked_question_answer_pairs(tmp_path):
    root = find_funsd_root(write_funsd(tmp_path))
    assert root == tmp_path / "dataset"
    cases = load_funsd(root)
    first = cases[0]
    assert first.id == "funsd-form_a" and first.kind == "funsd" and first.pages == 1
    assert first.truth == {"q_date": "03/14/1999", "q_to": "Jordan Lee"}
    assert first.schema.field("q_date").labels[0] == "Date"
    assert "q_fax_number" in first.asked  # a decoy borrowed from the other form
    assert "q_pages" not in first.asked  # a question without an answer is not scored
    assert first.data.startswith(b"\x89PNG") and "FAX COVER SHEET" in first.source_text


def test_find_funsd_root_explains_what_it_needs(tmp_path):
    with pytest.raises(DatasetError, match="testing_data"):
        find_funsd_root(tmp_path)


class FakeFetcher:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []

    def fetch(self, url):
        self.urls.append(url)
        return Fetched(url, url, self.data, "zip", "application/zip", ["datasets.example.com"])


def test_download_refuses_to_put_a_dataset_inside_the_repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "datasets").mkdir(parents=True)
    with pytest.raises(DatasetError, match="outside"):
        download_funsd(repo / "datasets", fetcher=FakeFetcher(b""), repo_root=repo)


def test_download_checks_the_hash_and_extracts_safely(tmp_path):
    data = zip_bytes({"dataset/testing_data/annotations/x.json": b'{"form": []}',
                      "dataset/testing_data/images/x.png": b"png"})
    fetcher = FakeFetcher(data)
    with pytest.raises(DatasetError, match="sha256"):
        download_funsd(tmp_path / "d1", fetcher=fetcher, sha256="0" * 64)
    root = download_funsd(tmp_path / "d2", fetcher=fetcher,
                          sha256=hashlib.sha256(data).hexdigest())
    assert root == (tmp_path / "d2" / "funsd" / "dataset").resolve()
    assert fetcher.urls == [FUNSD_URL, FUNSD_URL]
