"""Tests for sentiment/sources_stocks.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentiment.sources_stocks import (
    StocksSources,
    _fetch_fear_greed,
    _fetch_yahoo_news,
    _fetch_yahoo_price,
    fetch,
    to_prompt_text,
)


# ---------------------------------------------------------------------------
# _fetch_fear_greed
# ---------------------------------------------------------------------------

_CNN_FG_RESPONSE = {
    "fear_and_greed": {
        "score": 38.7,
        "rating": "Fear",
        "timestamp": "2024-01-01T00:00:00",
    },
    "fear_and_greed_historical": {},
}


class TestFetchFearGreed:
    def test_parses_valid_response(self):
        with patch("sentiment.sources_stocks._get_json", return_value=_CNN_FG_RESPONSE):
            value, label = _fetch_fear_greed()
        assert value == 39  # rounded
        assert label == "Fear"

    def test_rounds_score(self):
        data = {
            "fear_and_greed": {"score": 72.5, "rating": "Greed"},
        }
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            value, label = _fetch_fear_greed()
        assert value == 73
        assert label == "Greed"

    def test_returns_none_on_network_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            value, label = _fetch_fear_greed()
        assert value is None
        assert label is None

    def test_returns_none_on_missing_key(self):
        with patch("sentiment.sources_stocks._get_json", return_value={"other": {}}):
            value, label = _fetch_fear_greed()
        assert value is None
        assert label is None

    def test_returns_none_on_wrong_type(self):
        with patch("sentiment.sources_stocks._get_json", return_value=[1, 2, 3]):
            value, label = _fetch_fear_greed()
        assert value is None
        assert label is None


# ---------------------------------------------------------------------------
# _fetch_yahoo_price
# ---------------------------------------------------------------------------

def _yahoo_chart(closes: list[float | None]) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "indicators": {
                        "quote": [{"close": closes}]
                    }
                }
            ]
        }
    }


class TestFetchYahooPrice:
    def test_computes_24h_change(self):
        data = _yahoo_chart([100.0, 105.0])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch7d = _fetch_yahoo_price("AAPL")
        assert ch24 == pytest.approx(5.0)
        assert ch7d is None

    def test_computes_7d_change_with_8_closes(self):
        # 8 closes: base = closes[-8] = closes[0] = 100
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 110.0]
        data = _yahoo_chart(closes)
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch7d = _fetch_yahoo_price("AAPL")
        assert ch24 == pytest.approx((110.0 - 106.0) / 106.0 * 100)
        assert ch7d == pytest.approx(10.0)

    def test_skips_trailing_none(self):
        closes = [100.0, 105.0, None]
        data = _yahoo_chart(closes)
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch7d = _fetch_yahoo_price("AAPL")
        assert ch24 == pytest.approx(5.0)

    def test_returns_nones_on_network_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            assert _fetch_yahoo_price("AAPL") == (None, None)

    def test_returns_nones_on_single_close(self):
        data = _yahoo_chart([100.0])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            assert _fetch_yahoo_price("AAPL") == (None, None)

    def test_returns_nones_on_empty_closes(self):
        data = _yahoo_chart([])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            assert _fetch_yahoo_price("AAPL") == (None, None)

    def test_negative_24h_change(self):
        data = _yahoo_chart([110.0, 100.0])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, _ = _fetch_yahoo_price("AAPL")
        assert ch24 == pytest.approx(-100.0 / 110.0 * 100)


# ---------------------------------------------------------------------------
# _fetch_yahoo_news
# ---------------------------------------------------------------------------

_YAHOO_NEWS_RESPONSE = {
    "news": [
        {"title": "Apple beats earnings expectations"},
        {"title": "iPhone demand surges in Asia"},
        {"title": "Apple announces new MacBook"},
    ]
}


class TestFetchYahooNews:
    def test_returns_headlines(self):
        with patch("sentiment.sources_stocks._get_json", return_value=_YAHOO_NEWS_RESPONSE):
            headlines = _fetch_yahoo_news("AAPL")
        assert headlines == [
            "Apple beats earnings expectations",
            "iPhone demand surges in Asia",
            "Apple announces new MacBook",
        ]

    def test_caps_at_five(self):
        many = {"news": [{"title": f"Headline {i}"} for i in range(10)]}
        with patch("sentiment.sources_stocks._get_json", return_value=many):
            headlines = _fetch_yahoo_news("AAPL")
        assert len(headlines) == 5

    def test_returns_empty_on_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            assert _fetch_yahoo_news("AAPL") == []

    def test_skips_entries_without_title(self):
        data = {"news": [{"title": "Good headline"}, {"link": "no title here"}]}
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            headlines = _fetch_yahoo_news("AAPL")
        assert headlines == ["Good headline"]

    def test_returns_empty_on_missing_news_key(self):
        with patch("sentiment.sources_stocks._get_json", return_value={}):
            assert _fetch_yahoo_news("AAPL") == []


# ---------------------------------------------------------------------------
# fetch (integration)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_returns_stocks_sources(self):
        with (
            patch("sentiment.sources_stocks._fetch_fear_greed", return_value=(42, "Fear")),
            patch("sentiment.sources_stocks._fetch_yahoo_price", return_value=(1.5, -3.2)),
            patch("sentiment.sources_stocks._fetch_yahoo_news", return_value=["Big rally"]),
        ):
            result = fetch("AAPL")

        assert isinstance(result, StocksSources)
        assert result.symbol == "AAPL"
        assert result.fear_greed_value == 42
        assert result.fear_greed_label == "Fear"
        assert result.price_change_24h_pct == pytest.approx(1.5)
        assert result.price_change_7d_pct == pytest.approx(-3.2)
        assert result.news_headlines == ["Big rally"]
        assert result.fetched_at > 0

    def test_partial_failure_still_returns_object(self):
        with (
            patch("sentiment.sources_stocks._fetch_fear_greed", return_value=(None, None)),
            patch("sentiment.sources_stocks._fetch_yahoo_price", return_value=(None, None)),
            patch("sentiment.sources_stocks._fetch_yahoo_news", return_value=[]),
        ):
            result = fetch("MSFT")

        assert isinstance(result, StocksSources)
        assert result.symbol == "MSFT"
        assert result.fear_greed_value is None
        assert result.news_headlines == []


# ---------------------------------------------------------------------------
# to_prompt_text
# ---------------------------------------------------------------------------

class TestToPromptText:
    def _full_sources(self) -> StocksSources:
        return StocksSources(
            symbol="AAPL",
            fetched_at=1_700_000_000,
            fear_greed_value=42,
            fear_greed_label="Fear",
            price_change_24h_pct=2.5,
            price_change_7d_pct=-1.8,
            news_headlines=["Apple beats earnings", "iPhone demand surges"],
        )

    def test_includes_symbol(self):
        text = to_prompt_text(self._full_sources())
        assert "AAPL" in text

    def test_includes_fear_greed(self):
        text = to_prompt_text(self._full_sources())
        assert "42/100" in text
        assert "Fear" in text

    def test_includes_price_changes(self):
        text = to_prompt_text(self._full_sources())
        assert "+2.50%" in text
        assert "-1.80%" in text

    def test_includes_headlines(self):
        text = to_prompt_text(self._full_sources())
        assert "Apple beats earnings" in text
        assert "iPhone demand surges" in text

    def test_empty_sources_minimal_output(self):
        sources = StocksSources(symbol="NVDA", fetched_at=1_700_000_000)
        text = to_prompt_text(sources)
        assert "NVDA" in text
        assert "Fear" not in text
        assert "%" not in text

    def test_positive_sign_on_positive_changes(self):
        sources = StocksSources(
            symbol="MSFT",
            fetched_at=1_700_000_000,
            price_change_24h_pct=3.0,
        )
        text = to_prompt_text(sources)
        assert "+3.00%" in text

    def test_no_double_sign_on_negative_changes(self):
        sources = StocksSources(
            symbol="MSFT",
            fetched_at=1_700_000_000,
            price_change_24h_pct=-2.0,
        )
        text = to_prompt_text(sources)
        assert "-2.00%" in text
        assert "+-" not in text

    def test_cnn_label_in_output(self):
        text = to_prompt_text(self._full_sources())
        assert "CNN Fear & Greed" in text
