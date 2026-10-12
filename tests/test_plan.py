import pytest

from privdoc.config import Settings
from privdoc.errors import PrivacyViolation, UsageError
from privdoc.plan import build_plan
from privdoc.privacy import Tier
from tests.fakes import FakeTransport

LLM = {"PRIVDOC_LLM_BASE_URL": "https://llm.example.com/v1", "PRIVDOC_LLM_API_KEY": "k",
       "PRIVDOC_LLM_MODEL": "small", "PRIVDOC_LLM_ESCALATION_MODEL": "large"}
GCV = {"PRIVDOC_GCV_API_KEY": "g", "PRIVDOC_GCV_ENDPOINT": "https://ocr.example.com"}


def plan(env, tier, *, tesseract=True, **kw):
    settings = Settings.from_env(env)
    return build_plan(settings, tier, transport=FakeTransport(),
                      which=lambda cmd: "/usr/bin/tesseract" if tesseract else None, **kw)


def ids(items):
    return [item.id for item in items]


def test_local_tier_uses_only_tesseract_and_rules_even_with_keys_configured():
    p = plan({**LLM, **GCV}, Tier.LOCAL)
    assert ids(p.readers) == ["tesseract:eng:psm3:300dpi", "tesseract:eng:psm6:400dpi+clean"]
    assert ids(p.extractors) == ["rules"]
    assert p.gate.allowed_hosts == []
    assert all(r.destination is None for r in p.readers)


def test_local_tier_without_tesseract_has_no_ocr_and_says_how_to_fix_it():
    p = plan({}, Tier.LOCAL, tesseract=False)
    assert p.readers == [] and any("Tesseract" in note for note in p.notes)


def test_own_cloud_uses_your_cloud_ocr_and_skips_a_third_party_llm():
    p = plan({**LLM, **GCV}, Tier.OWN_CLOUD)
    assert ids(p.readers) == ["cloud-ocr:ocr.example.com"]
    assert ids(p.extractors) == ["rules"]
    assert p.gate.allowed_hosts == ["ocr.example.com"]
    assert any("llm.example.com" in note for note in p.notes)


def test_own_cloud_may_use_an_llm_you_host_yourself():
    p = plan({**LLM, **GCV, "PRIVDOC_LLM_OWNED": "1"}, Tier.OWN_CLOUD)
    assert ids(p.readers) == ["cloud-ocr:ocr.example.com", "vision-llm:large"]
    assert ids(p.extractors) == ["text-llm:small", "text-llm:large"]
    assert p.gate.allowed_hosts == ["llm.example.com", "ocr.example.com"]


def test_cloud_tier_reads_scans_with_a_vision_model_and_escalates_to_the_larger_one():
    p = plan({**LLM, **GCV}, Tier.CLOUD)
    assert ids(p.readers) == ["vision-llm:small", "vision-llm:large"]
    assert ids(p.extractors) == ["text-llm:small", "text-llm:large"]


def test_cloud_tier_falls_back_to_what_is_configured():
    assert ids(plan(GCV, Tier.CLOUD).readers) == ["cloud-ocr:ocr.example.com"]
    assert ids(plan({}, Tier.CLOUD).readers)[0].startswith("tesseract")
    assert ids(plan({}, Tier.CLOUD).extractors) == ["rules"]


def test_tesseract_escalates_to_a_vision_model_when_the_tier_allows_it():
    p = plan(LLM, Tier.CLOUD, engine="tesseract")
    assert ids(p.readers) == ["tesseract:eng:psm3:300dpi", "tesseract:eng:psm6:400dpi+clean",
                              "vision-llm:large"]


def test_no_duplicate_rung_when_the_escalation_model_is_the_same():
    p = plan({**LLM, "PRIVDOC_LLM_ESCALATION_MODEL": "small"}, Tier.CLOUD)
    assert ids(p.readers) == ["vision-llm:small"]


@pytest.mark.parametrize("env, tier, engine, extractor", [
    (LLM, Tier.LOCAL, "vision-llm", "auto"),
    (LLM, Tier.OWN_CLOUD, "vision-llm", "auto"),
    (GCV, Tier.LOCAL, "cloud-ocr", "auto"),
    (LLM, Tier.LOCAL, "auto", "llm"),
    (LLM, Tier.OWN_CLOUD, "auto", "llm"),
])
def test_explicitly_asking_for_a_disallowed_engine_is_refused(env, tier, engine, extractor):
    with pytest.raises(PrivacyViolation):
        plan(env, tier, engine=engine, extractor=extractor)


@pytest.mark.parametrize("env, engine, extractor, tesseract", [
    ({}, "vision-llm", "auto", True), ({}, "cloud-ocr", "auto", True),
    ({}, "tesseract", "auto", False), ({}, "auto", "llm", True), ({}, "laser", "auto", True),
])
def test_explicitly_asking_for_an_unconfigured_engine_is_a_usage_error(env, engine, extractor,
                                                                     tesseract):
    with pytest.raises(UsageError):
        plan(env, Tier.CLOUD, engine=engine, extractor=extractor, tesseract=tesseract)


def test_rules_can_be_forced_in_the_cloud_tier():
    assert ids(plan(LLM, Tier.CLOUD, extractor="rules").extractors) == ["rules"]


def test_description_for_the_check_command():
    described = plan({**LLM, **GCV}, Tier.OWN_CLOUD).describe()
    assert described["tier"] == "own-cloud"
    assert described["ocr"] == ["cloud-ocr:ocr.example.com"]
    assert described["destinations"] == ["Cloud Vision OCR (ocr.example.com, owner)"]


@pytest.mark.parametrize("found, expected", [("/opt/bin/tesseract", 2), (None, 0)])
def test_tesseract_is_looked_up_when_the_plan_is_built_not_at_import(monkeypatch, found,
                                                                      expected):
    monkeypatch.setattr("shutil.which", lambda cmd, *a, **k: found)
    plan = build_plan(Settings.from_env({}), Tier.LOCAL)
    assert len(plan.readers) == expected
