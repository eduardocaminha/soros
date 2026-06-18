"""Tests for sentiment/sources_stocks.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentiment.sources_stocks import (
    StockSources,
    _fetch_cnn_fear_greed,
    _fetch_yahoo_news,
    _fetch_yahoo_price,
    fetch,
    to_prompt_text,
)


# ---------------------------------------------------------------------------
# _fetch_cnn_fear_greed
# ---------------------------------------------------------------------------

_CNN_RESPONSE = {
    "fear_and_greed": {
        "score": 42.5,
        "rating": "fear",
        "timestamp": "2024-01-01 12:00:00.000000",
    }
}


class TestFetchCnnFearGreed:
    def test_parses_valid_response(self):
        with patch("sentiment.sources_stocks._get_json", return_value=_CNN_RESPONSE):
            value, label = _fetch_cnn_fear_greed()
        assert value == 42  # round(42.5) → 42 (banker's rounding)
        assert label == "Fear"

    def test_returns_none_on_network_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            value, label = _fetch_cnn_fear_greed()
        assert value is None
        assert label is None

    def test_returns_none_on_malformed_response(self):
        with patch("sentiment.sources_stocks._get_json", return_value={"other": {}}):
            value, label = _fetch_cnn_fear_greed()
        assert value is None
        assert label is None

    def test_title_cases_label(self):
        data = {"fear_and_greed": {"score": "75", "rating": "greed"}}
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            _, label = _fetch_cnn_fear_greed()
        assert label == "Greed"


# ---------------------------------------------------------------------------
# _fetch_yahoo_price
# ---------------------------------------------------------------------------

_YAHOO_CHART_RESPONSE = {
    "chart": {
        "result": [
            {
                "meta": {"regularMarketChangePercent": 1.5},
                "indicators": {
                    "quote": [{"close": [180.0, 182.0, 181.5, 183.0, 185.0]}]
                },
            }
        ],
        "error": None,
    }
}


class TestFetchYahooPrice:
    def test_parses_valid_response(self):
        with patch("sentiment.sources_stocks._get_json", return_value=_YAHOO_CHART_RESPONSE):
            ch24, ch5d = _fetch_yahoo_price("AAPL")
        assert ch24 == pytest.approx(1.5)
        # (185.0 - 180.0) / 180.0 * 100 ≈ 2.778
        assert ch5d == pytest.approx(2.777, rel=0.01)

    def test_returns_nones_on_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            assert _fetch_yahoo_price("AAPL") == (None, None)

    def test_returns_none_24h_when_meta_missing(self):
        data = {
            "chart": {
                "result": [
                    {
                        "meta": {},
                        "indicators": {"quote": [{"close": [100.0, 102.0]}]},
                    }
                ]
            }
        }
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch5d = _fetch_yahoo_price("AAPL")
        assert ch24 is None
        assert ch5d == pytest.approx(2.0)

    def test_returns_none_5d_when_fewer_than_two_closes(self):
        data = {
            "chart": {
                "result": [
                    {
                        "meta": {"regularMarketChangePercent": 0.5},
                        "indicators": {"quote": [{"close": [100.0]}]},
                    }
                ]
            }
        }
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch5d = _fetch_yahoo_price("AAPL")
        assert ch24 == pytest.approx(0.5)
        assert ch5d is None

    def test_skips_none_closes_for_5d(self):
        data = {
            "chart": {
                "result": [
                    {
                        "meta": {"regularMarketChangePercent": 1.0},
                        "indicators": {"quote": [{"close": [None, 100.0, None, 105.0]}]},
                    }
                ]
            }
        }
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            _, ch5d = _fetch_yahoo_price("AAPL")
        assert ch5d == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _fetch_yahoo_news
# ---------------------------------------------------------------------------

_YAHOO_NEWS_RESPONSE = {
    "news": [
        {"title": "Apple hits record high"},
        {"title": "Tech stocks rally"},
        {"title": "Fed holds rates steady"},
    ]
}


class TestFetchYahooNews:
    def test_returns_headlines(self):
        with patch("sentiment.sources_stocks._get_json", return_value=_YAHOO_NEWS_RESPONSE):
            headlines = _fetch_yahoo_news("AAPL")
        assert headlines == [
            "Apple hits record high",
            "Tech stocks rally",
            "Fed holds rates steady",
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


# ---------------------------------------------------------------------------
# fetch (integration of all sources)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_returns_stock_sources(self):
        with (
            patch("sentiment.sources_stocks._fetch_cnn_fear_greed", return_value=(45, "Fear")),
            patch("sentiment.sources_stocks._fetch_yahoo_price", return_value=(1.5, 3.2)),
            patch("sentiment.sources_stocks._fetch_yahoo_news", return_value=["Headline A"]),
        ):
            result = fetch("AAPL")

        assert isinstance(result, StockSources)
        assert result.symbol == "AAPL"
        assert result.fear_greed_value == 45
        assert result.fear_greed_label == "Fear"
        assert result.price_change_24h_pct == pytest.approx(1.5)
        assert result.price_change_5d_pct == pytest.approx(3.2)
        assert result.news_headlines == ["Headline A"]
        assert result.fetched_at > 0

    def test_partial_failure_still_returns_object(self):
        with (
            patch("sentiment.sources_stocks._fetch_cnn_fear_greed", return_value=(None, None)),
            patch("sentiment.sources_stocks._fetch_yahoo_price", return_value=(None, None)),
            patch("sentiment.sources_stocks._fetch_yahoo_news", return_value=[]),
        ):
            result = fetch("MSFT")

        assert isinstance(result, StockSources)
        assert result.symbol == "MSFT"
        assert result.fear_greed_value is None
        assert result.news_headlines == []


# ---------------------------------------------------------------------------
# to_prompt_text
# ---------------------------------------------------------------------------

class TestToPromptText:
    def _full_sources(self) -> StockSources:
        return StockSources(
            symbol="AAPL",
            fetched_at=1_700_000_000,
            fear_greed_value=38,
            fear_greed_label="Fear",
            price_change_24h_pct=2.1,
            price_change_5d_pct=-1.3,
            news_headlines=["Apple AI event", "Earnings beat"],
        )

    def test_includes_symbol(self):
        text = to_prompt_text(self._full_sources())
        assert "AAPL" in text

    def test_includes_fear_greed(self):
        text = to_prompt_text(self._full_sources())
        assert "38/100" in text
        assert "Fear" in text

    def test_includes_price_changes(self):
        text = to_prompt_text(self._full_sources())
        assert "+2.10%" in text
        assert "-1.30%" in text

    def test_includes_headlines(self):
        text = to_prompt_text(self._full_sources())
        assert "Apple AI event" in text
        assert "Earnings beat" in text

    def test_empty_sources_minimal_output(self):
        sources = StockSources(symbol="NVDA", fetched_at=1_700_000_000)
        text = to_prompt_text(sources)
        assert "NVDA" in text
        assert "%" not in text
        assert "Fear" not in text

    def test_positive_sign_on_positive_24h(self):
        sources = StockSources(
            symbol="MSFT", fetched_at=1_700_000_000, price_change_24h_pct=3.5
        )
        text = to_prompt_text(sources)
        assert "+3.50%" in text

    def test_no_double_sign_on_negative(self):
        sources = StockSources(
            symbol="MSFT", fetched_at=1_700_000_000, price_change_24h_pct=-2.0
        )
        text = to_prompt_text(sources)
        assert "-2.00%" in text
        assert "+-" not in text
