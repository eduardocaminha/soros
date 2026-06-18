"""Tests for sentiment/sources_stocks.py."""

from __future__ import annotations

from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest

from sentiment.sources_stocks import (
    StockSources,
    _ticker,
    _fetch_vix,
    _fetch_yf_quote,
    _fetch_news,
    fetch,
    to_prompt_text,
)


# ---------------------------------------------------------------------------
# _ticker
# ---------------------------------------------------------------------------

class TestTicker:
    def test_plain_symbol(self):
        assert _ticker("AAPL") == "AAPL"

    def test_symbol_with_exchange(self):
        assert _ticker("AAPL:NASDAQ") == "AAPL"

    def test_lowercase(self):
        assert _ticker("msft") == "MSFT"


# ---------------------------------------------------------------------------
# _fetch_yf_quote
# ---------------------------------------------------------------------------

def _yf_response(closes: list[float], volumes: list[int]) -> dict:
    return {
        "chart": {
            "result": [{
                "meta": {},
                "indicators": {
                    "quote": [{"close": closes, "volume": volumes}]
                },
            }]
        }
    }


class TestFetchYfQuote:
    def test_parses_24h_change(self):
        data = _yf_response([100.0, 110.0], [1_000_000, 1_200_000])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch5d, vol = _fetch_yf_quote("AAPL")
        assert ch24 == pytest.approx(10.0)
        assert ch5d is None  # not enough data points for 5d
        assert vol is None   # need at least 2 volumes where first is avg base

    def test_parses_5d_change(self):
        closes = [80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 100.0]
        volumes = [1_000_000] * 7
        data = _yf_response(closes, volumes)
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, ch5d, vol = _fetch_yf_quote("AAPL")
        # 24h: (100 - 90) / 90 * 100
        assert ch24 == pytest.approx((100.0 - 90.0) / 90.0 * 100)
        # 5d: closes[-1]=100, closes[-6]=82 → (100-82)/82*100
        assert ch5d == pytest.approx((100.0 - 82.0) / 82.0 * 100)

    def test_volume_ratio(self):
        closes = [100.0, 101.0, 102.0]
        volumes = [1_000_000, 1_000_000, 2_000_000]
        data = _yf_response(closes, volumes)
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            _, _, vol = _fetch_yf_quote("AAPL")
        # avg of first 2 = 1_000_000; last = 2_000_000 → ratio = 2.0
        assert vol == pytest.approx(2.0)

    def test_returns_nones_on_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            assert _fetch_yf_quote("AAPL") == (None, None, None)

    def test_returns_nones_on_too_few_closes(self):
        data = _yf_response([100.0], [1_000_000])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            assert _fetch_yf_quote("AAPL") == (None, None, None)

    def test_filters_none_values(self):
        data = _yf_response([None, 100.0, None, 110.0], [None, 1_000_000, None, 1_200_000])
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            ch24, _, _ = _fetch_yf_quote("AAPL")
        assert ch24 == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _fetch_vix
# ---------------------------------------------------------------------------

_VIX_RESPONSE = {
    "chart": {
        "result": [{
            "indicators": {
                "quote": [{"close": [18.5, 19.2]}]
            }
        }]
    }
}


class TestFetchVix:
    def test_returns_latest_close(self):
        with patch("sentiment.sources_stocks._get_json", return_value=_VIX_RESPONSE):
            vix = _fetch_vix()
        assert vix == pytest.approx(19.2)

    def test_returns_none_on_failure(self):
        with patch("sentiment.sources_stocks._get_json", return_value=None):
            assert _fetch_vix() is None

    def test_returns_none_on_empty_closes(self):
        data = {"chart": {"result": [{"indicators": {"quote": [{"close": []}]}}]}}
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            assert _fetch_vix() is None

    def test_skips_none_closes(self):
        data = {"chart": {"result": [{"indicators": {"quote": [{"close": [None, 20.0]}]}}]}}
        with patch("sentiment.sources_stocks._get_json", return_value=data):
            assert _fetch_vix() == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# _fetch_news
# ---------------------------------------------------------------------------

_RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Yahoo Finance</title>
    <item><title>Apple beats earnings</title></item>
    <item><title>iPhone sales surge</title></item>
    <item><title>Tim Cook speaks on AI</title></item>
  </channel>
</rss>"""


class TestFetchNews:
    def test_returns_headlines_from_rss(self):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.read.return_value = _RSS_XML
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("sentiment.sources_stocks.urlopen", return_value=mock_resp):
            headlines = _fetch_news("AAPL")

        assert headlines == ["Apple beats earnings", "iPhone sales surge", "Tim Cook speaks on AI"]

    def test_caps_at_five(self):
        items = "".join(f"<item><title>Headline {i}</title></item>" for i in range(10))
        xml = f"""<?xml version="1.0"?><rss><channel>{items}</channel></rss>""".encode()

        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.read.return_value = xml
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("sentiment.sources_stocks.urlopen", return_value=mock_resp):
            headlines = _fetch_news("AAPL")

        assert len(headlines) == 5

    def test_returns_empty_on_network_error(self):
        from urllib.error import URLError
        with patch("sentiment.sources_stocks.urlopen", side_effect=URLError("err")):
            assert _fetch_news("AAPL") == []

    def test_returns_empty_on_parse_error(self):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not xml at all <<<>>>"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("sentiment.sources_stocks.urlopen", return_value=mock_resp):
            assert _fetch_news("AAPL") == []

    def test_skips_items_without_title(self):
        xml = b"""<?xml version="1.0"?><rss><channel>
            <item><title>Good headline</title></item>
            <item><link>no title</link></item>
        </channel></rss>"""

        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.read.return_value = xml
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("sentiment.sources_stocks.urlopen", return_value=mock_resp):
            headlines = _fetch_news("AAPL")

        assert headlines == ["Good headline"]


# ---------------------------------------------------------------------------
# fetch (integration of all sources)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_returns_stock_sources(self):
        with (
            patch("sentiment.sources_stocks._fetch_yf_quote", return_value=(1.5, -2.0, 1.3)),
            patch("sentiment.sources_stocks._fetch_vix", return_value=18.5),
            patch("sentiment.sources_stocks._fetch_news", return_value=["AAPL up"]),
        ):
            result = fetch("AAPL")

        assert isinstance(result, StockSources)
        assert result.symbol == "AAPL"
        assert result.price_change_24h_pct == pytest.approx(1.5)
        assert result.price_change_5d_pct == pytest.approx(-2.0)
        assert result.volume_ratio == pytest.approx(1.3)
        assert result.vix_value == pytest.approx(18.5)
        assert result.news_headlines == ["AAPL up"]
        assert result.fetched_at > 0

    def test_strips_exchange_suffix(self):
        with (
            patch("sentiment.sources_stocks._fetch_yf_quote", return_value=(None, None, None)) as mock_yf,
            patch("sentiment.sources_stocks._fetch_vix", return_value=None),
            patch("sentiment.sources_stocks._fetch_news", return_value=[]),
        ):
            fetch("MSFT:NASDAQ")

        mock_yf.assert_called_once_with("MSFT")

    def test_partial_failure_still_returns_object(self):
        with (
            patch("sentiment.sources_stocks._fetch_yf_quote", return_value=(None, None, None)),
            patch("sentiment.sources_stocks._fetch_vix", return_value=None),
            patch("sentiment.sources_stocks._fetch_news", return_value=[]),
        ):
            result = fetch("TSLA")

        assert isinstance(result, StockSources)
        assert result.symbol == "TSLA"
        assert result.price_change_24h_pct is None
        assert result.news_headlines == []


# ---------------------------------------------------------------------------
# to_prompt_text
# ---------------------------------------------------------------------------

class TestToPromptText:
    def _full_sources(self) -> StockSources:
        return StockSources(
            symbol="AAPL",
            fetched_at=1_700_000_000,
            price_change_24h_pct=2.5,
            price_change_5d_pct=-1.0,
            volume_ratio=1.75,
            vix_value=18.3,
            news_headlines=["Apple hits ATH", "iPhone demand strong"],
        )

    def test_includes_symbol(self):
        text = to_prompt_text(self._full_sources())
        assert "AAPL" in text

    def test_includes_price_changes(self):
        text = to_prompt_text(self._full_sources())
        assert "+2.50%" in text
        assert "-1.00%" in text

    def test_includes_volume_ratio(self):
        text = to_prompt_text(self._full_sources())
        assert "1.75x" in text

    def test_includes_vix(self):
        text = to_prompt_text(self._full_sources())
        assert "18.3" in text

    def test_includes_headlines(self):
        text = to_prompt_text(self._full_sources())
        assert "Apple hits ATH" in text
        assert "iPhone demand strong" in text

    def test_empty_sources_minimal_output(self):
        sources = StockSources(symbol="GOOG", fetched_at=1_700_000_000)
        text = to_prompt_text(sources)
        assert "GOOG" in text
        assert "%" not in text
        assert "VIX" not in text

    def test_positive_sign_on_positive_changes(self):
        sources = StockSources(
            symbol="NVDA",
            fetched_at=1_700_000_000,
            price_change_24h_pct=5.0,
        )
        text = to_prompt_text(sources)
        assert "+5.00%" in text

    def test_no_double_sign_on_negative(self):
        sources = StockSources(
            symbol="NVDA",
            fetched_at=1_700_000_000,
            price_change_24h_pct=-3.0,
        )
        text = to_prompt_text(sources)
        assert "-3.00%" in text
        assert "+-" not in text
