"""Cost estimates. Prices change, so none are hard-coded: configure the ones you pay."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import UsageError


@dataclass(frozen=True)
class Price:
    input_per_mtok: float  # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens


def parse_prices(spec: str) -> dict[str, Price]:
    """'model-a=0.15/0.60,model-b=2.5/10' (USD per 1M input/output tokens)."""
    prices: dict[str, Price] = {}
    for item in filter(None, (part.strip() for part in (spec or "").split(","))):
        name, _, rates = item.partition("=")
        inp, _, out = rates.partition("/")
        try:
            if not name.strip() or not out:
                raise ValueError
            prices[name.strip()] = Price(float(inp), float(out))
        except ValueError:
            raise UsageError(f"bad price {item!r}; expected model=INPUT/OUTPUT in USD per 1M "
                             "tokens") from None
    return prices


@dataclass(frozen=True)
class Prices:
    models: dict[str, Price] = field(default_factory=dict)
    ocr_per_page: float | None = None

    def llm_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.models.get(model)
        if price is None:
            return None
        return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6

    def ocr_cost(self, pages: int) -> float | None:
        return None if self.ocr_per_page is None else pages * self.ocr_per_page
