"""Settings from environment variables (and an optional .env file).

The default tier is ``local``: nothing leaves the machine unless you opt in, even when API keys
are present in the environment. Bad values fail loudly and name the variable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .errors import UsageError
from .pricing import Prices, parse_prices
from .privacy import Tier, Trust, parse_tier

OPENAI_BASE = "https://api.openai.com/v1"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def default_cache_dir(env: Mapping[str, str]) -> Path:
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "privdoc"
    if os.name == "nt" and env.get("LOCALAPPDATA"):
        return Path(env["LOCALAPPDATA"]) / "privdoc" / "cache"
    home = Path(env["HOME"]) if env.get("HOME") else Path.home()
    return home / ".cache" / "privdoc"


@dataclass(frozen=True)
class Settings:
    tier: Tier = Tier.LOCAL
    llm_base_url: str = OPENAI_BASE
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_escalation_model: str = ""
    llm_trust: Trust = Trust.THIRD_PARTY
    llm_max_px: int = 2000
    llm_temperature: float | None = None
    gcv_endpoint: str = "https://vision.googleapis.com"
    gcv_api_key: str = field(default="", repr=False)
    gcv_access_token: str = field(default="", repr=False)
    gcv_quota_project: str = ""
    ocr_lang: str = "en"
    tesseract_cmd: str = "tesseract"
    tesseract_lang: str = "eng"
    cache_dir: Path | None = None
    cache_ttl_days: float = 30.0
    prices: Prices = field(default_factory=Prices)
    max_pages: int = 20
    max_escalations: int = 3
    max_file_mb: float = 50.0
    min_text_chars: int = 30
    scan_cover: float = 0.6
    concurrency: int = 4
    timeout_s: float = 300.0
    fetch_timeout_s: float = 30.0
    fetch_max_mb: float = 25.0
    date_order: str = "MDY"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        r = _Reader(env)
        base_url = r.text("PRIVDOC_LLM_BASE_URL", OPENAI_BASE)
        api_key = r.text("PRIVDOC_LLM_API_KEY", "")
        if not api_key and urlsplit(base_url).hostname == "api.openai.com":
            # never hand an OpenAI key to some other server just because it is in the env
            api_key = r.text("OPENAI_API_KEY", "")
        cache_on = r.flag("PRIVDOC_CACHE", True)
        cache_dir = Path(r.text("PRIVDOC_CACHE_DIR", "")) if r.text("PRIVDOC_CACHE_DIR", "") \
            else default_cache_dir(env)
        date_order = r.text("PRIVDOC_DATE_ORDER", "MDY").upper()
        if date_order not in ("MDY", "DMY"):
            raise UsageError("PRIVDOC_DATE_ORDER must be MDY or DMY")
        temperature = r.text("PRIVDOC_LLM_TEMPERATURE", "")
        ocr_price = r.text("PRIVDOC_OCR_PRICE_PER_PAGE", "")
        return cls(
            tier=parse_tier(r.text("PRIVDOC_TIER", "local")),
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_model=r.text("PRIVDOC_LLM_MODEL", ""),
            llm_escalation_model=r.text("PRIVDOC_LLM_ESCALATION_MODEL", ""),
            llm_trust=Trust.OWNER if r.flag("PRIVDOC_LLM_OWNED", False) else Trust.THIRD_PARTY,
            llm_max_px=r.integer("PRIVDOC_LLM_MAX_PX", 2000, 256, 10000),
            llm_temperature=r.number("PRIVDOC_LLM_TEMPERATURE", 0.0, 0.0, 2.0)
            if temperature else None,
            gcv_endpoint=r.text("PRIVDOC_GCV_ENDPOINT", "https://vision.googleapis.com"),
            gcv_api_key=r.text("PRIVDOC_GCV_API_KEY", ""),
            gcv_access_token=r.text("PRIVDOC_GCV_ACCESS_TOKEN", ""),
            gcv_quota_project=r.text("PRIVDOC_GCV_QUOTA_PROJECT", ""),
            ocr_lang=r.text("PRIVDOC_OCR_LANG", "en"),
            tesseract_cmd=r.text("PRIVDOC_TESSERACT_CMD", "tesseract"),
            tesseract_lang=r.text("PRIVDOC_TESSERACT_LANG", "eng"),
            cache_dir=cache_dir if cache_on else None,
            cache_ttl_days=r.number("PRIVDOC_CACHE_TTL_DAYS", 30.0, 0.0, 3650.0),
            prices=Prices(models=parse_prices(r.text("PRIVDOC_PRICES", "")),
                          ocr_per_page=r.number("PRIVDOC_OCR_PRICE_PER_PAGE", 0.0, 0.0, 10.0)
                          if ocr_price else None),
            max_pages=r.integer("PRIVDOC_MAX_PAGES", 20, 1, 1000),
            max_escalations=r.integer("PRIVDOC_MAX_ESCALATIONS", 3, 0, 100),
            max_file_mb=r.number("PRIVDOC_MAX_FILE_MB", 50.0, 0.1, 2048.0),
            min_text_chars=r.integer("PRIVDOC_MIN_TEXT_CHARS", 30, 0, 10000),
            scan_cover=r.number("PRIVDOC_SCAN_COVER", 0.6, 0.0, 1.01),
            concurrency=r.integer("PRIVDOC_CONCURRENCY", 4, 1, 16),
            timeout_s=r.number("PRIVDOC_TIMEOUT", 300.0, 1.0, 7200.0),
            fetch_timeout_s=r.number("PRIVDOC_FETCH_TIMEOUT", 30.0, 1.0, 600.0),
            fetch_max_mb=r.number("PRIVDOC_FETCH_MAX_MB", 25.0, 0.1, 2048.0),
            date_order=date_order,
        )


class _Reader:
    def __init__(self, env: Mapping[str, str]) -> None:
        self.env = env

    def text(self, name: str, default: str) -> str:
        return (self.env.get(name) or default).strip()

    def flag(self, name: str, default: bool) -> bool:
        raw = self.env.get(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
        raise UsageError(f"{name} must be on/off (got {raw!r})")

    def number(self, name: str, default: float, lo: float, hi: float) -> float:
        raw = self.env.get(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            raise UsageError(f"{name} must be a number (got {raw!r})") from None
        if not lo <= value <= hi:
            raise UsageError(f"{name} must be between {lo:g} and {hi:g} (got {raw})")
        return value

    def integer(self, name: str, default: int, lo: int, hi: int) -> int:
        raw = self.env.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise UsageError(f"{name} must be a whole number (got {raw!r})") from None
        if not lo <= value <= hi:
            raise UsageError(f"{name} must be between {lo} and {hi} (got {raw})")
        return value


def read_env_file(path: str | Path) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments, blank lines, 'export ' prefixes and quotes are handled."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or " " in key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values
