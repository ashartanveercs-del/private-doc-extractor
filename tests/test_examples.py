"""The examples and the dataset script are part of the product: keep them working."""

import importlib.util
import io
import json
import runpy
import sys
import zipfile
from pathlib import Path

import pytest

from privdoc.fetch import Fetched

ROOT = Path(__file__).resolve().parents[1]


def run_example(name: str, out: Path, monkeypatch, capsys) -> str:
    monkeypatch.setattr(sys, "argv", [name, "--out", str(out)])
    monkeypatch.setattr("shutil.which", lambda cmd, *a, **k: None)  # behave as if no Tesseract
    runpy.run_path(str(ROOT / "examples" / name), run_name="__main__")
    return capsys.readouterr().out


def test_quickstart_reads_the_typed_invoice_and_reports_the_unread_scan(tmp_path, monkeypatch,
                                                                        capsys):
    out = run_example("quickstart.py", tmp_path, monkeypatch, capsys)
    assert "invoice-42-text.pdf: valid=True complete=True" in out
    assert "invoice-42-scan.pdf: valid=False complete=False" in out
    assert "unread page 1: scanned page, and no OCR engine is available" in out
    assert "sent_to=nowhere" in out


def test_custom_schema_example_produces_schema_valid_json(tmp_path, monkeypatch, capsys):
    out = run_example("custom_schema.py", tmp_path, monkeypatch, capsys)
    result = json.loads(out)
    assert result["valid"] is True
    assert result["data"] == {
        "ticket_id": "TCK-04217", "opened": "2026-03-02", "customer": "Casey Nguyen",
        "email": "casey@example.com", "priority": "high",
        "device": "Commuter bike, 8-speed", "warranty": True, "estimate": 149.0}


def load_script():
    spec = importlib.util.spec_from_file_location("download_funsd",
                                                  ROOT / "scripts" / "download_funsd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZipFetcher:
    def fetch(self, url):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("dataset/testing_data/annotations/a.json", '{"form": []}')
        return Fetched(url, url, buf.getvalue(), "zip", "application/zip", ["data.example.com"])


def test_download_script_unpacks_outside_the_repo(tmp_path, capsys):
    script = load_script()
    assert script.main(["--dest", str(tmp_path)], fetcher=ZipFetcher()) == 0
    assert "privdoc bench --dataset funsd" in capsys.readouterr().out
    assert (tmp_path / "funsd" / "dataset" / "testing_data").is_dir()


@pytest.mark.parametrize("inside", ["datasets", "src"])
def test_download_script_refuses_the_repository(capsys, inside):
    script = load_script()
    code = script.main(["--dest", str(ROOT / inside)], fetcher=ZipFetcher())
    assert code != 0 and "outside" in capsys.readouterr().err
    assert not (ROOT / inside / "funsd").exists()
