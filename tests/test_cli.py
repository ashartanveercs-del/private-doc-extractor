import json

import pytest

from privdoc import __version__
from privdoc.cache import Cache, cache_key
from privdoc.cli import main
from privdoc.synth import make_doc, render
from tests.fakes import FakeTesseract, FakeTransport
from tests.oracle import page_texts

BASE_ENV = {"PRIVDOC_CACHE": "off", "PRIVDOC_CONCURRENCY": "1"}


def fakes(tesseract=None, found=True):
    return {"tesseract_runner": tesseract or FakeTesseract(), "transport": FakeTransport(),
            "which": lambda cmd: "/usr/bin/tesseract" if found else None}


def run(capsys, argv, env=None, **pipeline_kwargs):
    code = main(argv, env={**BASE_ENV, **(env or {})}, pipeline_kwargs=pipeline_kwargs or fakes())
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def invoice_pdf(tmp_path):
    doc = make_doc("invoice", 3)
    path = tmp_path / "invoice.pdf"
    path.write_bytes(render(doc, "text").data)
    return doc, path


def test_extract_prints_json_and_exits_0_when_valid(capsys, invoice_pdf):
    doc, path = invoice_pdf
    code, out, err = run(capsys, ["extract", str(path), "--schema", "invoice", "--tier", "local"])
    result = json.loads(out)
    assert code == 0 and result["valid"] and result["tier"] == "local"
    assert result["data"]["invoice_number"] == doc.truth["invoice_number"]
    assert result["source"]["name"] == "invoice.pdf" and err == ""


def test_incomplete_results_exit_1_but_still_print_everything(capsys, tmp_path):
    path = tmp_path / "receipt.pdf"
    path.write_bytes(render(make_doc("receipt", 4), "mixed").data)
    code, out, _ = run(capsys, ["extract", str(path), "--schema", "receipt"],
                       **fakes(found=False))
    result = json.loads(out)
    assert code == 1 and not result["valid"] and result["pages_unread"][0]["page"] == 2


@pytest.mark.parametrize("argv, env, message", [
    (["--tier", "secret"], {}, "unknown privacy tier"),
    (["--schema", "nope.json"], {}, "cannot read schema"),
    (["--engine", "vision-llm"], {"PRIVDOC_LLM_MODEL": "small"}, "local tier"),
    (["--pages", "9"], {}, "outside"),
])
def test_usage_and_privacy_errors_exit_2_with_one_calm_line(capsys, invoice_pdf, argv, env,
                                                            message):
    _, path = invoice_pdf
    full = ["extract", str(path), "--schema", "invoice", *argv]
    if "--schema" in argv:
        full = ["extract", str(path), *argv]
    code, out, err = run(capsys, full, env)
    assert code == 2 and out == ""
    assert err.startswith("privdoc: error:") and message in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_unreadable_documents_exit_3(capsys, tmp_path):
    path = tmp_path / "page.pdf"
    path.write_bytes(b"<html><body>Please sign in</body></html>")
    code, out, err = run(capsys, ["extract", str(path), "--schema", "invoice"])
    assert code == 3 and "not a supported document" in err


def test_out_file_and_compact_output(capsys, invoice_pdf, tmp_path):
    _, path = invoice_pdf
    target = tmp_path / "result.json"
    code, out, err = run(capsys, ["extract", str(path), "--schema", "invoice", "--out",
                                  str(target)])
    assert code == 0 and out == "" and "result.json" in err
    assert json.loads(target.read_text(encoding="utf-8"))["valid"]
    code, out, _ = run(capsys, ["extract", str(path), "--schema", "invoice", "--compact"])
    assert len(out.strip().splitlines()) == 1


def test_tier_and_settings_come_from_the_environment_and_env_files(capsys, invoice_pdf,
                                                                    tmp_path):
    _, path = invoice_pdf
    env_file = tmp_path / ".env"
    env_file.write_text("PRIVDOC_TIER=cloud\nPRIVDOC_MAX_PAGES=1\n", encoding="utf-8")
    code, out, _ = run(capsys, ["extract", str(path), "--schema", "invoice", "--env-file",
                                str(env_file)], {"PRIVDOC_MAX_PAGES": "5"})
    result = json.loads(out)
    assert result["tier"] == "cloud"
    assert result["complete"]  # the real environment (5 pages) wins over the file (1 page)


def test_the_cli_local_tier_never_touches_the_network(capsys, tmp_path, no_network_at_all):
    doc = make_doc("tax_form", 1)
    path = tmp_path / "form.pdf"
    path.write_bytes(render(doc, "scan").data)
    tesseract = FakeTesseract(pages={1: page_texts(doc, "scan")[0]})
    code, out, _ = run(capsys, ["extract", str(path), "--schema", "tax_form"],
                       {"PRIVDOC_LLM_MODEL": "small", "OPENAI_API_KEY": "sk-unused"},
                       **fakes(tesseract))
    assert code == 0 and json.loads(out)["data_sent_to"] == []
    assert no_network_at_all.attempts == []


def test_check_describes_every_tier_without_sending_anything(capsys, no_network_at_all):
    code, out, _ = run(capsys, ["check"], {"PRIVDOC_LLM_MODEL": "small"}, **fakes(found=False))
    report = json.loads(out)
    assert code == 0 and report["default_tier"] == "local" and report["tesseract"] is None
    assert set(report["tiers"]) == {"cloud", "own-cloud", "local"}
    assert report["tiers"]["local"]["ocr"] == []
    assert report["tiers"]["cloud"]["ocr"] == ["vision-llm:small"]
    assert no_network_at_all.attempts == []


def test_synth_writes_documents_with_their_ground_truth(capsys, tmp_path):
    code, out, _ = run(capsys, ["synth", str(tmp_path / "out"), "--kind", "receipt",
                                "--variant", "text", "--variant", "photo", "--count", "2"])
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert code == 0 and files == [
        "receipt-0-photo.jpg", "receipt-0-photo.truth.json", "receipt-0-text.pdf",
        "receipt-0-text.truth.json", "receipt-1-photo.jpg", "receipt-1-photo.truth.json",
        "receipt-1-text.pdf", "receipt-1-text.truth.json"]
    truth = json.loads((tmp_path / "out" / "receipt-1-text.truth.json").read_text("utf-8"))
    assert truth["truth"] == make_doc("receipt", 1).truth and truth["variant"] == "text"


def test_cache_purge(capsys, tmp_path):
    cache = Cache(tmp_path)
    for i in range(2):
        cache.put(cache_key(i=i), {"i": i})
    code, out, _ = run(capsys, ["cache", "purge"], {"PRIVDOC_CACHE": "on",
                                                     "PRIVDOC_CACHE_DIR": str(tmp_path)})
    assert code == 0 and "2" in out and not list(tmp_path.rglob("*.json"))


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"], env=BASE_ENV)
    assert info.value.code == 0 and __version__ in capsys.readouterr().out
