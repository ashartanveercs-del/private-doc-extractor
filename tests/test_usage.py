import pytest

from privdoc.errors import UsageError
from privdoc.pricing import Price, Prices, parse_prices
from privdoc.records import PageReading, Usage


def test_prices_parse_from_a_compact_spec():
    prices = parse_prices("model-a=0.15/0.60, model-b=2.5/10")
    assert prices == {"model-a": Price(0.15, 0.60), "model-b": Price(2.5, 10.0)}
    assert parse_prices("") == {}


@pytest.mark.parametrize("spec", ["model-a", "model-a=0.15", "model-a=x/1", "=1/2"])
def test_bad_price_specs_are_usage_errors(spec):
    with pytest.raises(UsageError):
        parse_prices(spec)


def test_llm_cost_is_tokens_times_price_per_million():
    prices = Prices(models={"model-a": Price(0.15, 0.60)})
    assert prices.llm_cost("model-a", 1000, 200) == pytest.approx(0.00027)
    assert prices.llm_cost("unknown-model", 1000, 200) is None


def test_ocr_cost_is_per_page_when_a_price_is_configured():
    assert Prices(ocr_per_page=0.0015).ocr_cost(2) == pytest.approx(0.003)
    assert Prices().ocr_cost(2) is None


def test_usage_adds_up_and_unknown_cost_stays_unknown():
    a = Usage(llm_calls=1, input_tokens=100, output_tokens=10, cost_usd=0.01)
    b = Usage(ocr_pages=1, cost_usd=0.002)
    total = a + b
    assert (total.llm_calls, total.input_tokens, total.ocr_pages) == (1, 100, 1)
    assert total.cost_usd == pytest.approx(0.012)
    assert (a + Usage(llm_calls=1, cost_usd=None)).cost_usd is None
    assert Usage().cost_usd == 0.0


def test_page_readings_round_trip_through_the_cache_format():
    reading = PageReading(page=2, reader="tesseract:eng:psm3:300dpi", text="Total: 5.00",
                          legibility="fair", fields=None,
                          usage=Usage(ocr_pages=1, cost_usd=0.0), seconds=1.25)
    again = PageReading.from_cache(reading.to_cache())
    assert again.cached and again.text == "Total: 5.00" and again.legibility == "fair"
    assert again.reader == reading.reader and again.usage == reading.usage
