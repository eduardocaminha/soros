"""Tests for sentiment/sources_crypto.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentiment.sources_crypto import (
    CryptoSources,
    _base,
    _fetch_coingecko,
    _fetch_fear_greed,
    _fetch_news,
    fetch,
    to_prompt_text,
)


# ---------------------------------------------------------------------------
# _base
# ---------------------------------------------------------------------------

class TestBase:
    def test_extracts_base_currency(self):
        assert _base("BTC/USDT") == "BTC"

    def test_uppercase(self):
        assert _base("eth/usdt") == "ETH"

    def test_no_slash(self):
        assert _base("BTC") == "BTC"


# ---------------------------------------------------------------------------
# _fetch_fear_greed
# ---------------------------------------------------------------------------

_FNG_RESPONSE = {
    "data": [{"value": "25", "value_classification": "Fear"}],
    "metadata": {"error": None},
}


class TestFetchFearGreed:
    def test_parses_valid_response(self):
        with patch("sentiment.sources_crypto._get_json", return_value=_FNG_RESPONSE):
            value, label = _fetch_fear_greed()
        assert value == 25
        assert label == "Fear"

    def test_returns_none_on_network_failure(self):
        with patch("sentiment.sources_crypto._get_json", return_value=None):
            value, label = _fetch_fear_greed()
        assert value is None
        assert label is None

    def test_returns_none_on_malformed_response(self):
        with patch("sentiment.sources_crypto._get_json", return_value={"data": []}):
            value, label = _fetch_fear_greed()
        assert value is None
        assert label is None


# ---------------------------------------------------------------------------
# _fetch_coingecko
# ---------------------------------------------------------------------------

_COINGECKO_RESPONSE = [
    {
        "id": "bitcoin",
        "price_change_percentage_24h": 2.5,
        "price_change_percentage_7d_in_currency": -3.1,
        "market_cap_rank": 1,
    }
]


class TestFetchCoingecko:
    def test_parses_valid_response(self):
        with patch("sentiment.sources_crypto._get_json", return_value=_COINGECKO_RESPONSE):
            ch24, ch7d, rank = _fetch_coingecko("BTC")
        assert ch24 == pytest.approx(2.5)
        assert ch7d == pytest.approx(-3.1)
        assert rank == 1

    def test_returns_nones_on_failure(self):
        with patch("sentiment.sources_crypto._get_json", return_value=None):
            assert _fetch_coingecko("BTC") == (None, None, None)

    def test_returns_nones_on_empty_list(self):
        with patch("sentiment.sources_crypto._get_json", return_value=[]):
            assert _fetch_coingecko("BTC") == (None, None, None)

    def test_uses_id_map_for_known_symbol(self):
        captured = []

        def fake_get_json(url: str):
            captured.append(url)
            return []

        with patch("sentiment.sources_crypto._get_json", side_effect=fake_get_json):
            _fetch_coingecko("ETH")

        assert "ethereum" in captured[0]

    def test_lowercases_unknown_symbol(self):
        captured = []

        def fake_get_json(url: str):
            captured.append(url)
            return []

        with patch("sentiment.sources_crypto._get_json", side_effect=fake_get_json):
            _fetch_coingecko("PEPE")

        assert "pepe" in captured[0]


# ---------------------------------------------------------------------------
# _fetch_news
# ---------------------------------------------------------------------------

_NEWS_RESPONSE = {
    "Data": [
        {"title": "Bitcoin hits new ATH"},
        {"title": "Crypto market rebounds"},
        {"title": "Regulation news"},
    ]
}


class TestFetchNews:
    def test_returns_headlines(self):
        with patch("sentiment.sources_crypto._get_json", return_value=_NEWS_RESPONSE):
            headlines = _fetch_news("BTC")
        assert headlines == ["Bitcoin hits new ATH", "Crypto market rebounds", "Regulation news"]

    def test_caps_at_five(self):
        many = {"Data": [{"title": f"Headline {i}"} for i in range(10)]}
        with patch("sentiment.sources_crypto._get_json", return_value=many):
            headlines = _fetch_news("BTC")
        assert len(headlines) == 5

    def test_returns_empty_on_failure(self):
        with patch("sentiment.sources_crypto._get_json", return_value=None):
            assert _fetch_news("BTC") == []

    def test_skips_entries_without_title(self):
        data = {"Data": [{"title": "Good headline"}, {"body": "no title here"}]}
        with patch("sentiment.sources_crypto._get_json", return_value=data):
            headlines = _fetch_news("BTC")
        assert headlines == ["Good headline"]


# ---------------------------------------------------------------------------
# fetch (integration of all sources)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_returns_crypto_sources(self):
        with (
            patch("sentiment.sources_crypto._fetch_fear_greed", return_value=(60, "Greed")),
            patch("sentiment.sources_crypto._fetch_coingecko", return_value=(1.5, -2.0, 1)),
            patch("sentiment.sources_crypto._fetch_news", return_value=["Headline A"]),
        ):
            result = fetch("BTC/USDT")

        assert isinstance(result, CryptoSources)
        assert result.symbol == "BTC/USDT"
        assert result.fear_greed_value == 60
        assert result.fear_greed_label == "Greed"
        assert result.price_change_24h_pct == pytest.approx(1.5)
        assert result.price_change_7d_pct == pytest.approx(-2.0)
        assert result.market_cap_rank == 1
        assert result.news_headlines == ["Headline A"]
        assert result.fetched_at > 0

    def test_partial_failure_still_returns_object(self):
        with (
            patch("sentiment.sources_crypto._fetch_fear_greed", return_value=(None, None)),
            patch("sentiment.sources_crypto._fetch_coingecko", return_value=(None, None, None)),
            patch("sentiment.sources_crypto._fetch_news", return_value=[]),
        ):
            result = fetch("ETH/USDT")

        assert isinstance(result, CryptoSources)
        assert result.symbol == "ETH/USDT"
        assert result.fear_greed_value is None
        assert result.news_headlines == []


# ---------------------------------------------------------------------------
# to_prompt_text
# ---------------------------------------------------------------------------

class TestToPromptText:
    def _full_sources(self) -> CryptoSources:
        return CryptoSources(
            symbol="BTC/USDT",
            fetched_at=1_700_000_000,
            fear_greed_value=72,
            fear_greed_label="Greed",
            price_change_24h_pct=3.14,
            price_change_7d_pct=-1.5,
            market_cap_rank=1,
            news_headlines=["Big BTC rally", "ETF approved"],
        )

    def test_includes_symbol(self):
        text = to_prompt_text(self._full_sources())
        assert "BTC/USDT" in text

    def test_includes_fear_greed(self):
        text = to_prompt_text(self._full_sources())
        assert "72/100" in text
        assert "Greed" in text

    def test_includes_price_changes(self):
        text = to_prompt_text(self._full_sources())
        assert "+3.14%" in text
        assert "-1.50%" in text

    def test_includes_market_cap_rank(self):
        text = to_prompt_text(self._full_sources())
        assert "#1" in text

    def test_includes_headlines(self):
        text = to_prompt_text(self._full_sources())
        assert "Big BTC rally" in text
        assert "ETF approved" in text

    def test_empty_sources_minimal_output(self):
        sources = CryptoSources(symbol="SOL/USDT", fetched_at=1_700_000_000)
        text = to_prompt_text(sources)
        assert "SOL/USDT" in text
        assert "Fear" not in text
        assert "%" not in text

    def test_positive_sign_on_positive_changes(self):
        sources = CryptoSources(
            symbol="ETH/USDT",
            fetched_at=1_700_000_000,
            price_change_24h_pct=5.0,
        )
        text = to_prompt_text(sources)
        assert "+5.00%" in text

    def test_no_sign_on_negative_changes(self):
        sources = CryptoSources(
            symbol="ETH/USDT",
            fetched_at=1_700_000_000,
            price_change_24h_pct=-2.0,
        )
        text = to_prompt_text(sources)
        assert "-2.00%" in text
        assert "+-" not in text
