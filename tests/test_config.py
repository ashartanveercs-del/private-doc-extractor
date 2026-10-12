from pathlib import Path

import pytest

from privdoc.config import Settings, read_env_file
from privdoc.errors import UsageError
from privdoc.pricing import Price
from privdoc.privacy import Tier, Trust


def test_defaults_are_private_by_default():
    s = Settings.from_env({"HOME": "/home/someone"})
    assert s.tier is Tier.LOCAL
    assert s.llm_model == "" and s.llm_api_key == "" and s.gcv_api_key == ""
    assert s.cache_dir is not None and s.cache_dir.name == "privdoc"


def test_everything_is_configurable_from_the_environment(tmp_path):
    s = Settings.from_env({
        "PRIVDOC_TIER": "own_cloud", "PRIVDOC_LLM_BASE_URL": "https://llm.internal.example.com/v1",
        "PRIVDOC_LLM_API_KEY": "k1", "PRIVDOC_LLM_MODEL": "small",
        "PRIVDOC_LLM_ESCALATION_MODEL": "large", "PRIVDOC_LLM_OWNED": "yes",
        "PRIVDOC_GCV_API_KEY": "g1", "PRIVDOC_GCV_ENDPOINT": "https://ocr.example.com",
        "PRIVDOC_TESSERACT_CMD": "/opt/tess/bin/tesseract", "PRIVDOC_TESSERACT_LANG": "eng+deu",
        "PRIVDOC_CACHE_DIR": str(tmp_path), "PRIVDOC_PRICES": "small=0.1/0.4",
        "PRIVDOC_OCR_PRICE_PER_PAGE": "0.0015", "PRIVDOC_MAX_PAGES": "5",
        "PRIVDOC_MAX_ESCALATIONS": "1", "PRIVDOC_SCAN_COVER": "0.8",
        "PRIVDOC_DATE_ORDER": "dmy", "PRIVDOC_CONCURRENCY": "2",
    })
    assert s.tier is Tier.OWN_CLOUD and s.llm_trust is Trust.OWNER
    assert (s.llm_model, s.llm_escalation_model, s.llm_api_key) == ("small", "large", "k1")
    assert s.gcv_endpoint == "https://ocr.example.com" and s.gcv_api_key == "g1"
    assert s.tesseract_cmd == "/opt/tess/bin/tesseract" and s.tesseract_lang == "eng+deu"
    assert s.cache_dir == Path(tmp_path)
    assert s.prices.models == {"small": Price(0.1, 0.4)} and s.prices.ocr_per_page == 0.0015
    assert (s.max_pages, s.max_escalations, s.scan_cover, s.date_order, s.concurrency) == (
        5, 1, 0.8, "DMY", 2)


def test_the_openai_key_is_only_used_for_openai_itself():
    official = Settings.from_env({"OPENAI_API_KEY": "sk-1", "PRIVDOC_LLM_MODEL": "m"})
    assert official.llm_api_key == "sk-1"
    elsewhere = Settings.from_env({"OPENAI_API_KEY": "sk-1", "PRIVDOC_LLM_MODEL": "m",
                                   "PRIVDOC_LLM_BASE_URL": "https://other.example.com/v1"})
    assert elsewhere.llm_api_key == ""


@pytest.mark.parametrize("env, name", [
    ({"PRIVDOC_TIER": "secret"}, "tier"), ({"PRIVDOC_MAX_PAGES": "lots"}, "PRIVDOC_MAX_PAGES"),
    ({"PRIVDOC_MAX_PAGES": "0"}, "PRIVDOC_MAX_PAGES"),
    ({"PRIVDOC_LLM_OWNED": "maybe"}, "PRIVDOC_LLM_OWNED"),
    ({"PRIVDOC_DATE_ORDER": "YMD"}, "PRIVDOC_DATE_ORDER"),
    ({"PRIVDOC_SCAN_COVER": "2"}, "PRIVDOC_SCAN_COVER"),
])
def test_bad_values_fail_loudly_and_name_the_variable(env, name):
    with pytest.raises(UsageError, match=name):
        Settings.from_env(env)


def test_cache_can_be_switched_off():
    assert Settings.from_env({"PRIVDOC_CACHE": "off"}).cache_dir is None


def test_secrets_never_appear_in_repr():
    s = Settings.from_env({"PRIVDOC_LLM_API_KEY": "sk-secret-1", "PRIVDOC_GCV_API_KEY": "g-2",
                           "PRIVDOC_GCV_ACCESS_TOKEN": "ya29-3"})
    assert not {"sk-secret-1", "g-2", "ya29-3"} & set(repr(s).replace("'", " ").split())
    assert "sk-secret-1" not in repr(s)


def test_env_files(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\n\nexport PRIVDOC_TIER=cloud\nPRIVDOC_LLM_MODEL="small"\n'
                    "PRIVDOC_PRICES='small=0.1/0.4'\nNOT A LINE\n", encoding="utf-8")
    assert read_env_file(path) == {"PRIVDOC_TIER": "cloud", "PRIVDOC_LLM_MODEL": "small",
                                   "PRIVDOC_PRICES": "small=0.1/0.4"}


def test_the_example_env_file_parses_and_keeps_the_private_default():
    root = Path(__file__).resolve().parents[1]
    s = Settings.from_env(read_env_file(root / ".env.example"))
    assert s.tier is Tier.LOCAL and s.llm_model and s.llm_trust is Trust.THIRD_PARTY
