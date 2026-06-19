"""Tests for data/stocks_collector.py."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
from data.stocks_collector import (
    _alpaca_bars_to_candles,
    _alpaca_timeframe,
    _fetch_alpaca_bars,
    _fetch_yfinance_bars,
    _upsert_candles,
    _yfinance_interval,
    _yfinance_period,
    collect_once,
)


# ---------------------------------------------------------------------------
# Fixtures — isolated in-memory DB
# ---------------------------------------------------------------------------

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

def _json_resp(obj: dict) -> bytes:
    return json.dumps(obj).encode()


class _FakeResponse:
    """Context-manager compatible urllib response stub."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _mock_yf_dataframe(rows: list[tuple[int, float, float, float, float, float]]):
    """Return a mock object that behaves like a yfinance history DataFrame."""
    mock_df = MagicMock()
    mock_df.empty = len(rows) == 0
    mock_df.tail.return_value = mock_df

    items = []
    for ts, o, h, l, c, v in rows:
        mock_idx = MagicMock()
        mock_idx.timestamp.return_value = float(ts)
        mock_row = MagicMock()
        ohlcv = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}
        mock_row.__getitem__.side_effect = ohlcv.__getitem__
        items.append((mock_idx, mock_row))

    mock_df.iterrows.return_value = iter(items)
    return mock_df


_SAMPLE_BARS = [
    {"t": "2024-01-10T10:00:00Z", "o": 185.5, "h": 186.0, "l": 185.0, "c": 185.8, "v": 12345},
    {"t": "2024-01-10T11:00:00Z", "o": 185.8, "h": 187.0, "l": 185.5, "c": 186.5, "v": 9876},
]

_CANDLES = [
    (1_700_000_000, 185.0, 186.0, 184.5, 185.5, 10000.0),
    (1_700_003_600, 185.5, 187.0, 185.0, 186.5, 12000.0),
]


# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

class TestAlpacaTimeframe:
    def test_1h_maps_to_1Hour(self):
        assert _alpaca_timeframe("1h") == "1Hour"

    def test_1d_maps_to_1Day(self):
        assert _alpaca_timeframe("1d") == "1Day"

    def test_1m_maps_to_1Min(self):
        assert _alpaca_timeframe("1m") == "1Min"

    def test_unknown_defaults_to_1Hour(self):
        assert _alpaca_timeframe("xyz") == "1Hour"


class TestYfinanceInterval:
    def test_1h_maps_to_1h(self):
        assert _yfinance_interval("1h") == "1h"

    def test_4h_maps_to_1h(self):
        # yfinance has no 4h interval
        assert _yfinance_interval("4h") == "1h"

    def test_1w_maps_to_1wk(self):
        assert _yfinance_interval("1w") == "1wk"

    def test_unknown_defaults_to_1h(self):
        assert _yfinance_interval("xyz") == "1h"


class TestYfinancePeriod:
    def test_few_hourly_bars_fits_in_7d(self):
        assert _yfinance_period(5, "1h") == "7d"

    def test_200_hourly_bars(self):
        # 200 h ≈ 8.3 days → should be at least "7d"
        result = _yfinance_period(200, "1h")
        assert result in ("7d", "1mo", "2mo", "3mo", "6mo", "1y")

    def test_many_daily_bars_returns_1y(self):
        assert _yfinance_period(400, "1d") == "1y"


# ---------------------------------------------------------------------------
# _alpaca_bars_to_candles
# ---------------------------------------------------------------------------

class TestAlpacaBarsToCandles:
    def test_converts_iso_timestamp_to_unix_seconds(self):
        result = _alpaca_bars_to_candles(_SAMPLE_BARS)
        assert len(result) == 2
        ts = result[0][0]
        assert isinstance(ts, int)
        assert ts > 0

    def test_correct_ohlcv_values(self):
        result = _alpaca_bars_to_candles(_SAMPLE_BARS)
        ts, o, h, l, c, v = result[0]
        assert o == pytest.approx(185.5)
        assert h == pytest.approx(186.0)
        assert l == pytest.approx(185.0)
        assert c == pytest.approx(185.8)
        assert v == pytest.approx(12345)

    def test_skips_bar_with_bad_timestamp(self):
        bars = [{"t": "not-a-date", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 100}]
        assert _alpaca_bars_to_candles(bars) == []

    def test_skips_bar_missing_ohlcv_field(self):
        bars = [{"t": "2024-01-10T10:00:00Z", "o": 1.0}]
        assert _alpaca_bars_to_candles(bars) == []

    def test_empty_input_returns_empty(self):
        assert _alpaca_bars_to_candles([]) == []

    def test_timestamps_are_ascending(self):
        result = _alpaca_bars_to_candles(_SAMPLE_BARS)
        assert result[0][0] < result[1][0]


# ---------------------------------------------------------------------------
# _fetch_alpaca_bars
# ---------------------------------------------------------------------------

class TestFetchAlpacaBars:
    def test_returns_none_when_api_key_missing(self):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
        ):
            assert _fetch_alpaca_bars("AAPL") is None

    def test_returns_none_when_secret_missing(self):
        with (
            patch.object(config, "ALPACA_API_KEY", "key"),
            patch.object(config, "ALPACA_SECRET", ""),
        ):
            assert _fetch_alpaca_bars("AAPL") is None

    def test_returns_bars_on_success(self):
        with (
            patch.object(config, "ALPACA_API_KEY", "key"),
            patch.object(config, "ALPACA_SECRET", "secret"),
            patch("data.stocks_collector.urlopen",
                  return_value=_FakeResponse(_json_resp({"bars": _SAMPLE_BARS}))),
        ):
            result = _fetch_alpaca_bars("AAPL")
        assert result == _SAMPLE_BARS

    def test_returns_none_on_url_error(self):
        from urllib.error import URLError
        with (
            patch.object(config, "ALPACA_API_KEY", "key"),
            patch.object(config, "ALPACA_SECRET", "secret"),
            patch("data.stocks_collector.urlopen", side_effect=URLError("timeout")),
        ):
            assert _fetch_alpaca_bars("AAPL") is None

    def test_returns_empty_list_when_bars_key_absent(self):
        with (
            patch.object(config, "ALPACA_API_KEY", "key"),
            patch.object(config, "ALPACA_SECRET", "secret"),
            patch("data.stocks_collector.urlopen",
                  return_value=_FakeResponse(_json_resp({"next_page_token": None}))),
        ):
            assert _fetch_alpaca_bars("AAPL") == []


# ---------------------------------------------------------------------------
# _fetch_yfinance_bars
# ---------------------------------------------------------------------------

class TestFetchYfinanceBars:
    def test_returns_none_when_yfinance_not_installed(self):
        import builtins
        real_import = builtins.__import__

        def _block_yfinance(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("No module named 'yfinance'")
            return real_import(name, *args, **kwargs)

        # Remove yfinance from sys.modules cache so the ImportError fires
        sys.modules.pop("yfinance", None)
        with patch("builtins.__import__", side_effect=_block_yfinance):
            result = _fetch_yfinance_bars("AAPL")
        assert result is None

    def test_returns_candles_on_success(self):
        mock_df = _mock_yf_dataframe(_CANDLES)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = _fetch_yfinance_bars("AAPL")

        assert result is not None
        assert len(result) == 2
        ts, o, h, l, c, v = result[0]
        assert ts == 1_700_000_000
        assert o == pytest.approx(185.0)

    def test_returns_none_on_empty_dataframe(self):
        mock_df = _mock_yf_dataframe([])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = _fetch_yfinance_bars("AAPL")
        assert result is None

    def test_returns_none_on_ticker_exception(self):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("network error")
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = _fetch_yfinance_bars("AAPL")
        assert result is None


# ---------------------------------------------------------------------------
# _upsert_candles
# ---------------------------------------------------------------------------

class TestUpsertCandles:
    def test_inserts_candles(self, temp_db):
        count = _upsert_candles("AAPL", list(_CANDLES))
        assert count == 2

    def test_deduplication(self, temp_db):
        _upsert_candles("AAPL", [_CANDLES[0]])
        count = _upsert_candles("AAPL", [_CANDLES[0]])
        assert count == 0

    def test_asset_class_is_stocks(self, temp_db):
        _upsert_candles("AAPL", [_CANDLES[0]])
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT asset_class FROM prices WHERE symbol = 'AAPL'"
        ).fetchone()
        assert row[0] == "stocks"

    def test_funding_rate_is_null(self, temp_db):
        _upsert_candles("AAPL", [_CANDLES[0]])
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT funding_rate FROM prices WHERE symbol = 'AAPL'"
        ).fetchone()
        assert row[0] is None

    def test_empty_list_inserts_nothing(self, temp_db):
        assert _upsert_candles("AAPL", []) == 0


# ---------------------------------------------------------------------------
# collect_once
# ---------------------------------------------------------------------------

class TestCollectOnce:
    def test_uses_alpaca_when_credentials_set(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", "key"),
            patch.object(config, "ALPACA_SECRET", "secret"),
            patch("data.stocks_collector.urlopen",
                  return_value=_FakeResponse(_json_resp({"bars": _SAMPLE_BARS}))),
        ):
            results = collect_once(["AAPL"])
        assert results["AAPL"] > 0

    def test_falls_back_to_yfinance_when_no_credentials(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once(["AAPL"])
        assert results["AAPL"] == 2

    def test_falls_back_to_yfinance_when_alpaca_network_fails(self, temp_db):
        from urllib.error import URLError
        with (
            patch.object(config, "ALPACA_API_KEY", "key"),
            patch.object(config, "ALPACA_SECRET", "secret"),
            patch("data.stocks_collector.urlopen", side_effect=URLError("error")),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once(["AAPL"])
        assert results["AAPL"] == 2

    def test_returns_zero_when_both_sources_fail(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=None),
        ):
            results = collect_once(["AAPL"])
        assert results["AAPL"] == 0

    def test_uses_config_stock_symbols_by_default(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once()
        assert set(results.keys()) == set(config.STOCK_SYMBOLS)

    def test_multiple_symbols_all_succeed(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once(["AAPL", "MSFT"])
        assert results["AAPL"] == 2
        assert results["MSFT"] == 2

    def test_one_symbol_exception_does_not_abort_others(self, temp_db):
        call_count = 0

        def _side_effect(symbol, limit=None):
            nonlocal call_count
            call_count += 1
            if symbol == "AAPL":
                raise RuntimeError("unexpected error")
            return list(_CANDLES)

        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", side_effect=_side_effect),
        ):
            results = collect_once(["AAPL", "MSFT"])
        assert results["AAPL"] == 0
        assert results["MSFT"] == 2

    def test_watchlist_symbols_collected(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once(symbols=["AAPL"], watchlist=["TSLA"])
        assert "AAPL" in results
        assert "TSLA" in results

    def test_watchlist_symbol_in_pinned_collected_once(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once(symbols=["AAPL"], watchlist=["AAPL", "TSLA"])
        assert list(results.keys()).count("AAPL") == 1
        assert "TSLA" in results

    def test_watchlist_uses_short_limit(self, temp_db):
        captured: list[tuple] = []

        def _side_effect(symbol, limit=None):
            captured.append((symbol, limit))
            return list(_CANDLES)

        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch.object(config, "OHLCV_LIMIT", 200),
            patch.object(config, "WATCHLIST_OHLCV_LIMIT", 50),
            patch("data.stocks_collector._fetch_yfinance_bars", side_effect=_side_effect),
        ):
            collect_once(symbols=["AAPL"], watchlist=["TSLA"])

        aapl_limit = next(limit for sym, limit in captured if sym == "AAPL")
        tsla_limit = next(limit for sym, limit in captured if sym == "TSLA")
        assert aapl_limit == 200
        assert tsla_limit == 50

    def test_default_watchlist_from_config(self, temp_db):
        with (
            patch.object(config, "ALPACA_API_KEY", ""),
            patch.object(config, "ALPACA_SECRET", ""),
            patch.object(config, "STOCK_SYMBOLS", ["AAPL"]),
            patch.object(config, "STOCK_WATCHLIST", ["TSLA"]),
            patch("data.stocks_collector._fetch_yfinance_bars", return_value=list(_CANDLES)),
        ):
            results = collect_once()
        assert "AAPL" in results
        assert "TSLA" in results
