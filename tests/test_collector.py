"""Tests for data/collector.py (crypto OHLCV collection)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import config
import data.collector as collector_module
from data.collector import _upsert_candles, collect_once


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_perp_cache():
    """Clear the module-level perp-symbol cache between tests."""
    collector_module._perp_symbols_cache = None
    collector_module._perp_cache_ts = 0.0
    yield
    collector_module._perp_symbols_cache = None
    collector_module._perp_cache_ts = 0.0


class _FakeDB:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        schema = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()
        self._conn.executescript(schema)

    def connect(self) -> sqlite3.Connection:
        return self._conn


@pytest.fixture()
def temp_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    fake = _FakeDB(db_path)
    import database.db as db_module

    with patch.object(db_module, "_db", fake):
        yield db_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_candles(n: int = 5, ts_start: int = 1_700_000_000) -> list:
    """Return n ccxt-style candle rows [ts_ms, o, h, l, c, v]."""
    return [
        [int((ts_start + i * 3600) * 1000), 100.0, 101.0, 99.0, 100.5, 1000.0 + i]
        for i in range(n)
    ]


def _make_spot(candles_per_call: int = 5) -> MagicMock:
    mock = MagicMock()
    mock.fetch_ohlcv.return_value = _raw_candles(candles_per_call)
    return mock


def _make_perp_markets(*spot_symbols: str) -> dict:
    """Build a ccxt-style markets dict for the given spot symbols (all as USDT-M swaps)."""
    markets = {}
    for sym in spot_symbols:
        base = sym.split("/")[0]
        markets[f"{base}/USDT:USDT"] = {
            "base": base,
            "quote": "USDT",
            "settle": "USDT",
            "swap": True,
        }
    return markets


def _make_futures(perp_symbols: list[str] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.fetch_funding_rate.return_value = {"fundingRate": 0.0001}
    _syms = perp_symbols if perp_symbols is not None else ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    mock.load_markets.return_value = _make_perp_markets(*_syms)
    return mock


# ---------------------------------------------------------------------------
# _upsert_candles
# ---------------------------------------------------------------------------


class TestUpsertCandles:
    def test_inserts_candles(self, temp_db):
        candles = _raw_candles(3)
        count = _upsert_candles("BTC/USDT", candles, 0.0001)
        assert count == 3

    def test_deduplication(self, temp_db):
        candles = _raw_candles(2)
        _upsert_candles("BTC/USDT", candles, None)
        count = _upsert_candles("BTC/USDT", candles, None)
        assert count == 0

    def test_asset_class_is_crypto(self, temp_db):
        _upsert_candles("BTC/USDT", _raw_candles(1), None)
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT asset_class FROM prices WHERE symbol = 'BTC/USDT'"
        ).fetchone()
        assert row[0] == "crypto"

    def test_funding_rate_stored(self, temp_db):
        _upsert_candles("BTC/USDT", _raw_candles(1), 0.0001)
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT funding_rate FROM prices WHERE symbol = 'BTC/USDT'"
        ).fetchone()
        assert row[0] == pytest.approx(0.0001)

    def test_null_funding_rate(self, temp_db):
        _upsert_candles("BTC/USDT", _raw_candles(1), None)
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT funding_rate FROM prices WHERE symbol = 'BTC/USDT'"
        ).fetchone()
        assert row[0] is None


# ---------------------------------------------------------------------------
# collect_once
# ---------------------------------------------------------------------------


class TestCollectOnce:
    def test_defaults_to_crypto_symbols(self, temp_db):
        spot = _make_spot()
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            results = collect_once()
        assert set(results.keys()) == set(config.CRYPTO_SYMBOLS)

    def test_watchlist_symbols_also_collected(self, temp_db):
        spot = _make_spot()
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", ["XRP/USDT"]),
        ):
            results = collect_once()
        assert "BTC/USDT" in results
        assert "XRP/USDT" in results

    def test_symbol_in_both_collected_once(self, temp_db):
        """Symbol in both pinned and watchlist is only collected once (as pinned)."""
        spot = _make_spot()
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", ["BTC/USDT", "XRP/USDT"]),
        ):
            results = collect_once()
        assert list(results.keys()).count("BTC/USDT") == 1
        assert "XRP/USDT" in results

    def test_pinned_uses_full_ohlcv_limit(self, temp_db):
        spot = _make_spot()
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
            patch.object(config, "OHLCV_LIMIT", 200),
        ):
            collect_once()
        spot.fetch_ohlcv.assert_called_once_with(
            "BTC/USDT", timeframe=config.OHLCV_TIMEFRAME, limit=200
        )

    def test_watchlist_uses_short_limit(self, temp_db):
        spot = _make_spot()
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", ["XRP/USDT"]),
            patch.object(config, "OHLCV_LIMIT", 200),
            patch.object(config, "WATCHLIST_OHLCV_LIMIT", 50),
        ):
            collect_once()
        calls = spot.fetch_ohlcv.call_args_list
        btc_call = next(c for c in calls if c[0][0] == "BTC/USDT")
        xrp_call = next(c for c in calls if c[0][0] == "XRP/USDT")
        assert btc_call.kwargs["limit"] == 200
        assert xrp_call.kwargs["limit"] == 50

    def test_explicit_symbols_override(self, temp_db):
        spot = _make_spot()
        futures = _make_futures()
        with patch("data.collector._make_exchange", side_effect=[spot, futures]):
            results = collect_once(symbols=["ETH/USDT"], watchlist=[])
        assert set(results.keys()) == {"ETH/USDT"}

    def test_empty_watchlist_collects_only_pinned(self, temp_db):
        spot = _make_spot()
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            results = collect_once()
        assert set(results.keys()) == {"BTC/USDT"}

    def test_collection_failure_returns_zero_not_abort(self, temp_db):
        spot = MagicMock()
        futures = _make_futures()

        def _fail_btc(symbol, **kwargs):
            if symbol == "BTC/USDT":
                raise RuntimeError("network error")
            return _raw_candles(3)

        spot.fetch_ohlcv.side_effect = _fail_btc
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT", "ETH/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            results = collect_once()
        assert results["BTC/USDT"] == 0
        assert results["ETH/USDT"] > 0

    def test_inserted_count_reflects_new_rows(self, temp_db):
        spot = _make_spot(candles_per_call=3)
        futures = _make_futures()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            results = collect_once()
        assert results["BTC/USDT"] == 3


# ---------------------------------------------------------------------------
# Perp-symbol cache + no-perp silent skip
# ---------------------------------------------------------------------------


class TestFundingRateNoPerp:
    def test_no_warning_for_symbol_without_perp(self, temp_db, caplog):
        """Symbols not in the perp set must not log a WARNING."""
        # MATIC has no perp listed; BTC does
        futures = _make_futures(perp_symbols=["BTC/USDT"])
        spot = _make_spot()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT", "MATIC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
            caplog.at_level(logging.WARNING, logger="data.collector"),
        ):
            collect_once()
        warning_texts = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("MATIC/USDT" in t for t in warning_texts)

    def test_no_perp_symbol_gets_null_funding_rate(self, temp_db):
        """Symbols without a perp must be stored with funding_rate=None."""
        futures = _make_futures(perp_symbols=["BTC/USDT"])
        spot = _make_spot()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["MATIC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            collect_once()
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT funding_rate FROM prices WHERE symbol = 'MATIC/USDT' LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] is None

    def test_perp_symbol_still_fetches_funding_rate(self, temp_db):
        """Symbols with a perp continue to receive the real funding rate."""
        futures = _make_futures(perp_symbols=["BTC/USDT"])
        spot = _make_spot()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            collect_once()
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT funding_rate FROM prices WHERE symbol = 'BTC/USDT' LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.0001)
        futures.fetch_funding_rate.assert_called()

    def test_perp_fetch_not_called_for_no_perp_symbol(self, temp_db):
        """fetch_funding_rate must not be called for a symbol with no perp."""
        futures = _make_futures(perp_symbols=["BTC/USDT"])
        spot = _make_spot()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["MATIC/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            collect_once()
        futures.fetch_funding_rate.assert_not_called()

    def test_perp_cache_loaded_once_per_collect(self, temp_db):
        """load_markets should be called exactly once per collect_once call."""
        futures = _make_futures()
        spot = _make_spot()
        with (
            patch("data.collector._make_exchange", side_effect=[spot, futures]),
            patch.object(config, "CRYPTO_SYMBOLS", ["BTC/USDT", "ETH/USDT"]),
            patch.object(config, "CRYPTO_WATCHLIST", []),
        ):
            collect_once()
        futures.load_markets.assert_called_once()
