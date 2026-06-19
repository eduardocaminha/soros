"""Tests for data/universe.py — market-cap base tier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
import data.universe as universe_module
from data.universe import _EXCLUDE, get_base_universe, reset_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# _fetch_from_api (via get_base_universe with a mocked HTTP response)
# ---------------------------------------------------------------------------

def _make_api_response(tickers: list[str]) -> bytes:
    import json
    return json.dumps([{"symbol": t.lower(), "id": t.lower()} for t in tickers]).encode()


def _patch_urlopen(data: bytes):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.read.return_value = data
    return patch("urllib.request.urlopen", return_value=ctx)


class TestGetBaseUniverse:
    def test_returns_usdt_pairs(self):
        payload = _make_api_response(["BTC", "ETH", "SOL"])
        with _patch_urlopen(payload):
            result = get_base_universe(n=3)
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_filters_stablecoins(self):
        tickers = ["BTC", "USDT", "ETH", "USDC", "SOL"]
        payload = _make_api_response(tickers)
        with _patch_urlopen(payload):
            result = get_base_universe(n=3)
        assert "USDT/USDT" not in result
        assert "USDC/USDT" not in result
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_filters_wrapped_assets(self):
        tickers = ["BTC", "WBTC", "ETH", "WETH", "SOL"]
        payload = _make_api_response(tickers)
        with _patch_urlopen(payload):
            result = get_base_universe(n=3)
        assert "WBTC/USDT" not in result
        assert "WETH/USDT" not in result
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_respects_n_limit(self):
        tickers = ["BTC", "ETH", "SOL", "BNB", "XRP"]
        payload = _make_api_response(tickers)
        with _patch_urlopen(payload):
            result = get_base_universe(n=2)
        assert result == ["BTC/USDT", "ETH/USDT"]

    def test_uses_marketcap_top_n_config_default(self, monkeypatch):
        monkeypatch.setattr(config, "MARKETCAP_TOP_N", 2)
        payload = _make_api_response(["BTC", "ETH", "SOL"])
        with _patch_urlopen(payload):
            result = get_base_universe()
        assert len(result) == 2

    def test_result_is_uppercased(self):
        # CoinGecko returns lowercase symbols
        payload = _make_api_response(["btc", "eth"])
        with _patch_urlopen(payload):
            result = get_base_universe(n=2)
        assert result == ["BTC/USDT", "ETH/USDT"]


class TestCacheLogic:
    def test_cache_hit_skips_api(self):
        payload = _make_api_response(["BTC", "ETH"])
        with _patch_urlopen(payload) as mock_open:
            get_base_universe(n=2, refresh_secs=3600)
            get_base_universe(n=2, refresh_secs=3600)
        assert mock_open.call_count == 1

    def test_cache_miss_after_expiry(self):
        import time
        payload = _make_api_response(["BTC", "ETH"])
        with _patch_urlopen(payload) as mock_open:
            get_base_universe(n=2, refresh_secs=0)
            time.sleep(0.01)
            get_base_universe(n=2, refresh_secs=0)
        assert mock_open.call_count == 2

    def test_returns_cached_on_api_failure(self):
        payload = _make_api_response(["BTC", "ETH"])
        with _patch_urlopen(payload):
            first = get_base_universe(n=2, refresh_secs=0)

        # Second call with expired cache but broken API
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            second = get_base_universe(n=2, refresh_secs=0)

        assert second == first

    def test_returns_empty_on_first_failure(self):
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            result = get_base_universe(n=5, refresh_secs=3600)
        assert result == []

    def test_reset_cache_clears_state(self):
        payload = _make_api_response(["BTC", "ETH"])
        with _patch_urlopen(payload):
            get_base_universe(n=2, refresh_secs=3600)
        reset_cache()
        with _patch_urlopen(payload) as mock_open:
            get_base_universe(n=2, refresh_secs=3600)
        assert mock_open.call_count == 1


class TestExcludeSet:
    @pytest.mark.parametrize("ticker", ["USDT", "USDC", "DAI", "BUSD", "WBTC", "WETH", "STETH"])
    def test_known_exclusions_present(self, ticker: str):
        assert ticker in _EXCLUDE
