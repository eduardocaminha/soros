"""Tests for data/dex_scanner.py — DEX discovery signals."""

from __future__ import annotations

import json
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import config
import data.dex_scanner as dex_module
from data.dex_scanner import (
    _fetch_dexscreener,
    _fetch_geckoterminal,
    get_dex_trending_symbols,
    reset_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen(response_body: dict | list):
    """Return a context manager that yields a fake HTTP response."""
    payload = json.dumps(response_body).encode()

    class _Resp:
        def read(self):
            return payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    return _Resp()


def _dexscreener_boosts(addresses: list[str]) -> list[dict]:
    return [{"tokenAddress": a, "chainId": "ethereum"} for a in addresses]


def _dexscreener_pairs(symbol_by_address: dict[str, str]) -> dict:
    return {
        "pairs": [
            {"baseToken": {"symbol": sym, "address": addr}}
            for addr, sym in symbol_by_address.items()
        ]
    }


def _geckoterminal_response(symbols: list[str]) -> dict:
    data = []
    included = []
    for i, sym in enumerate(symbols):
        token_id = f"eth_0x{i:040x}"
        data.append({
            "id": f"pool_{i}",
            "type": "pool",
            "attributes": {"name": f"{sym} / USDT 0.3%"},
            "relationships": {
                "base_token": {"data": {"id": token_id, "type": "token"}}
            },
        })
        included.append({
            "id": token_id,
            "type": "token",
            "attributes": {"symbol": sym, "name": f"{sym} Token"},
        })
    return {"data": data, "included": included}


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# _fetch_dexscreener
# ---------------------------------------------------------------------------

class TestFetchDexscreener:
    def test_returns_symbols_from_pairs(self):
        boosts = _dexscreener_boosts(["0xABC", "0xDEF"])
        pairs = _dexscreener_pairs({"0xABC": "SOL", "0xDEF": "AVAX"})
        responses = [boosts, pairs]
        call_count = [0]

        def fake_urlopen(req, timeout=10):
            resp = _mock_urlopen(responses[call_count[0]])
            call_count[0] += 1
            return resp

        with patch("data.dex_scanner.urllib.request.urlopen", side_effect=fake_urlopen):
            result = _fetch_dexscreener()
        assert result == {"SOL", "AVAX"}

    def test_empty_boosts_returns_empty_set(self):
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen([])):
            result = _fetch_dexscreener()
        assert result == set()

    def test_symbols_uppercased(self):
        boosts = _dexscreener_boosts(["0xABC"])
        pairs = _dexscreener_pairs({"0xABC": "sol"})
        responses = [boosts, pairs]
        call_count = [0]

        def fake_urlopen(req, timeout=10):
            resp = _mock_urlopen(responses[call_count[0]])
            call_count[0] += 1
            return resp

        with patch("data.dex_scanner.urllib.request.urlopen", side_effect=fake_urlopen):
            result = _fetch_dexscreener()
        assert "SOL" in result

    def test_deduplicates_addresses(self):
        boosts = [
            {"tokenAddress": "0xABC", "chainId": "ethereum"},
            {"tokenAddress": "0xABC", "chainId": "bsc"},
        ]
        pairs = _dexscreener_pairs({"0xABC": "TOKEN"})
        responses = [boosts, pairs]
        call_count = [0]

        def fake_urlopen(req, timeout=10):
            # Verify batch URL has address only once
            if call_count[0] == 1:
                url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
                assert url.count("0xABC") == 1
            resp = _mock_urlopen(responses[call_count[0]])
            call_count[0] += 1
            return resp

        with patch("data.dex_scanner.urllib.request.urlopen", side_effect=fake_urlopen):
            _fetch_dexscreener()

    def test_missing_base_token_skipped(self):
        boosts = _dexscreener_boosts(["0xABC"])
        pairs_data = {"pairs": [{"dexId": "uniswap"}]}  # no baseToken
        responses = [boosts, pairs_data]
        call_count = [0]

        def fake_urlopen(req, timeout=10):
            resp = _mock_urlopen(responses[call_count[0]])
            call_count[0] += 1
            return resp

        with patch("data.dex_scanner.urllib.request.urlopen", side_effect=fake_urlopen):
            result = _fetch_dexscreener()
        assert result == set()

    def test_raises_on_network_error(self):
        with patch("data.dex_scanner.urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(OSError):
                _fetch_dexscreener()


# ---------------------------------------------------------------------------
# _fetch_geckoterminal
# ---------------------------------------------------------------------------

class TestFetchGeckoterminal:
    def test_returns_symbols_from_included_tokens(self):
        response = _geckoterminal_response(["PEPE", "SHIB", "DOGE"])
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = _fetch_geckoterminal()
        assert result == {"PEPE", "SHIB", "DOGE"}

    def test_symbols_uppercased(self):
        response = _geckoterminal_response(["pepe"])
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = _fetch_geckoterminal()
        assert "PEPE" in result

    def test_fallback_to_pool_name_when_no_included(self):
        response = {
            "data": [
                {
                    "id": "pool_1",
                    "type": "pool",
                    "attributes": {"name": "MEME / WETH 0.3%"},
                    "relationships": {"base_token": {"data": {"id": "unknown_id"}}},
                }
            ],
            "included": [],
        }
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = _fetch_geckoterminal()
        assert "MEME" in result

    def test_empty_response_returns_empty_set(self):
        response = {"data": [], "included": []}
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = _fetch_geckoterminal()
        assert result == set()

    def test_non_token_included_items_ignored(self):
        response = {
            "data": [],
            "included": [
                {"id": "x", "type": "network", "attributes": {"symbol": "FAKE"}}
            ],
        }
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = _fetch_geckoterminal()
        assert "FAKE" not in result

    def test_pool_name_without_slash_skipped(self):
        response = {
            "data": [
                {
                    "id": "pool_1",
                    "type": "pool",
                    "attributes": {"name": "NoSlashHere"},
                    "relationships": {},
                }
            ],
            "included": [],
        }
        with patch("data.dex_scanner.urllib.request.urlopen", return_value=_mock_urlopen(response)):
            result = _fetch_geckoterminal()
        assert result == set()

    def test_raises_on_network_error(self):
        with patch("data.dex_scanner.urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(OSError):
                _fetch_geckoterminal()


# ---------------------------------------------------------------------------
# get_dex_trending_symbols — caching and merging
# ---------------------------------------------------------------------------

class TestGetDexTrendingSymbols:
    def _patch_sources(self, dex_syms: set[str], gecko_syms: set[str]):
        """Context manager that stubs both source functions."""
        from unittest.mock import patch as _patch
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            with (
                _patch("data.dex_scanner._fetch_dexscreener", return_value=dex_syms),
                _patch("data.dex_scanner._fetch_geckoterminal", return_value=gecko_syms),
            ):
                yield

        return _ctx()

    def test_merges_both_sources(self):
        with self._patch_sources({"SOL", "AVAX"}, {"PEPE", "SHIB"}):
            result = get_dex_trending_symbols()
        assert result == frozenset({"SOL", "AVAX", "PEPE", "SHIB"})

    def test_returns_frozenset(self):
        with self._patch_sources({"SOL"}, set()):
            result = get_dex_trending_symbols()
        assert isinstance(result, frozenset)

    def test_cache_hit_skips_fetch(self):
        call_count = [0]
        def counting_dex():
            call_count[0] += 1
            return {"SOL"}

        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=counting_dex),
            patch("data.dex_scanner._fetch_geckoterminal", return_value=set()),
        ):
            get_dex_trending_symbols(cache_secs=60)
            get_dex_trending_symbols(cache_secs=60)
        assert call_count[0] == 1

    def test_cache_expires_after_secs(self):
        call_count = [0]
        def counting_dex():
            call_count[0] += 1
            return {"SOL"}

        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=counting_dex),
            patch("data.dex_scanner._fetch_geckoterminal", return_value=set()),
        ):
            get_dex_trending_symbols(cache_secs=1)
            # Force cache expiry by backdating the timestamp.
            dex_module._cache_ts -= 2
            get_dex_trending_symbols(cache_secs=1)
        assert call_count[0] == 2

    def test_dexscreener_failure_still_returns_geckoterminal(self):
        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=OSError("timeout")),
            patch("data.dex_scanner._fetch_geckoterminal", return_value={"PEPE"}),
        ):
            result = get_dex_trending_symbols()
        assert "PEPE" in result

    def test_geckoterminal_failure_still_returns_dexscreener(self):
        with (
            patch("data.dex_scanner._fetch_dexscreener", return_value={"SOL"}),
            patch("data.dex_scanner._fetch_geckoterminal", side_effect=OSError("timeout")),
        ):
            result = get_dex_trending_symbols()
        assert "SOL" in result

    def test_both_fail_returns_empty_frozenset(self):
        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=OSError("x")),
            patch("data.dex_scanner._fetch_geckoterminal", side_effect=OSError("y")),
        ):
            result = get_dex_trending_symbols()
        assert result == frozenset()

    def test_both_fail_rate_limits_retries(self):
        call_count = [0]
        def failing_dex():
            call_count[0] += 1
            raise OSError("x")

        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=failing_dex),
            patch("data.dex_scanner._fetch_geckoterminal", side_effect=OSError("y")),
        ):
            get_dex_trending_symbols(cache_secs=60)
            get_dex_trending_symbols(cache_secs=60)
        # Second call should be served from cache (cache_ts was set on first call).
        assert call_count[0] == 1

    def test_uses_config_cache_secs_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "DEX_SCAN_CACHE_SECS", 9999)
        call_count = [0]
        def counting_dex():
            call_count[0] += 1
            return {"X"}

        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=counting_dex),
            patch("data.dex_scanner._fetch_geckoterminal", return_value=set()),
        ):
            get_dex_trending_symbols()  # fetches, cache_ts set
            get_dex_trending_symbols()  # cache still valid (9999s)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# reset_cache
# ---------------------------------------------------------------------------

class TestResetCache:
    def test_reset_clears_symbols_and_timestamp(self):
        with (
            patch("data.dex_scanner._fetch_dexscreener", return_value={"SOL"}),
            patch("data.dex_scanner._fetch_geckoterminal", return_value=set()),
        ):
            get_dex_trending_symbols()
        assert dex_module._cache_ts > 0
        reset_cache()
        assert dex_module._cache_symbols == frozenset()
        assert dex_module._cache_ts == 0.0

    def test_reset_forces_refetch(self):
        call_count = [0]
        def counting_dex():
            call_count[0] += 1
            return {"X"}

        with (
            patch("data.dex_scanner._fetch_dexscreener", side_effect=counting_dex),
            patch("data.dex_scanner._fetch_geckoterminal", return_value=set()),
        ):
            get_dex_trending_symbols(cache_secs=999)
            reset_cache()
            get_dex_trending_symbols(cache_secs=999)
        assert call_count[0] == 2
