import privdoc
from privdoc.synth import make_doc, render


def test_one_call_api_reads_settings_from_the_environment(monkeypatch, tmp_path):
    for name in ("PRIVDOC_TIER", "PRIVDOC_LLM_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PRIVDOC_CACHE", "off")
    doc = make_doc("tax_form", 3)
    path = tmp_path / "statement.pdf"
    path.write_bytes(render(doc, "text").data)

    result = privdoc.extract(path, schema="tax_form")

    assert result.tier == "local" and result.valid
    assert result.data["payer_name"] == doc.truth["payer_name"]


def test_public_names():
    assert {"extract", "Pipeline", "ExtractOptions", "ExtractionResult", "Settings", "Tier",
            "load_schema"} <= set(privdoc.__all__)
