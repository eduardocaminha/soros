"""OHLCV collector for stocks — Alpaca v2 API (primary) + yfinance (fallback).

Primary path: uses urllib (stdlib) to call the Alpaca v2 Bars endpoint when
ALPACA_API_KEY + ALPACA_SECRET are both set in the environment.

Fallback path: imports yfinance and downloads via Yahoo Finance when Alpaca
credentials are absent or the Alpaca call fails for a symbol.

Candles are upserted into the prices table with asset_class='stocks'.
funding_rate is always NULL for stocks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config
from database.db import get_connection, get_logger

_log = get_logger(__name__)

_ALPACA_DATA_BASE = "https://data.alpaca.markets"
_TIMEOUT = 15  # seconds per HTTP request

# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

_TF_TO_ALPACA: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
    "1w": "1Week",
    "1M": "1Month",
}

_TF_TO_YFINANCE_INTERVAL: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",   # yfinance has no 4h; 1h is the closest
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
}

# Approximate hours per bar for each timeframe — used to estimate a yfinance period.
_TF_HOURS: dict[str, float] = {
    "1m": 1 / 60,
    "5m": 5 / 60,
    "15m": 15 / 60,
    "30m": 0.5,
    "1h": 1.0,
    "4h": 4.0,
    "1d": 24.0,
    "1w": 168.0,
    "1M": 720.0,
}


def _alpaca_timeframe(tf: str) -> str:
    return _TF_TO_ALPACA.get(tf, "1Hour")


def _yfinance_interval(tf: str) -> str:
    return _TF_TO_YFINANCE_INTERVAL.get(tf, "1h")


def _yfinance_period(limit: int, tf: str) -> str:
    """Return the smallest yfinance 'period' string that covers *limit* bars of *tf*."""
    hours = _TF_HOURS.get(tf, 1.0) * limit
    days = max(1, int(hours / 24) + 1)
    if days <= 7:
        return "7d"
    if days <= 30:
        return "1mo"
    if days <= 60:
        return "2mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    return "1y"


# ---------------------------------------------------------------------------
# Alpaca v2 data API
# ---------------------------------------------------------------------------

def _fetch_alpaca_bars(symbol: str, limit: int | None = None) -> list[dict[str, Any]] | None:
    """GET /v2/stocks/{symbol}/bars from Alpaca. Returns list of bar dicts or None."""
    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET):
        return None

    params = {
        "timeframe": _alpaca_timeframe(config.OHLCV_TIMEFRAME),
        "limit": limit if limit is not None else config.OHLCV_LIMIT,
        "adjustment": "raw",
        "feed": "iex",
    }
    url = f"{_ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars?{urlencode(params)}"
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET,
        "Accept": "application/json",
    }
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
        bars = data.get("bars") or []
        _log.debug("alpaca returned %d bars for %s", len(bars), symbol)
        return bars
    except (URLError, ValueError, OSError, TimeoutError) as exc:
        _log.warning("alpaca fetch failed for %s: %s", symbol, exc)
        return None


def _alpaca_bars_to_candles(
    bars: list[dict[str, Any]],
) -> list[tuple[int, float, float, float, float, float]]:
    """Convert Alpaca bar dicts to (ts_sec, open, high, low, close, volume) tuples."""
    result: list[tuple[int, float, float, float, float, float]] = []
    for bar in bars:
        try:
            ts = int(
                datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp()
            )
            result.append((ts, float(bar["o"]), float(bar["h"]), float(bar["l"]),
                           float(bar["c"]), float(bar["v"])))
        except (KeyError, ValueError, TypeError) as exc:
            _log.debug("skipping malformed alpaca bar: %s", exc)
    return result


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def _fetch_yfinance_bars(
    symbol: str,
    limit: int | None = None,
) -> list[tuple[int, float, float, float, float, float]] | None:
    """Download bars from yfinance. Returns (ts, o, h, l, c, v) tuples or None."""
    try:
        import yfinance as yf  # optional dependency
    except ImportError:
        _log.warning("yfinance not installed; cannot collect %s", symbol)
        return None

    effective_limit = limit if limit is not None else config.OHLCV_LIMIT
    interval = _yfinance_interval(config.OHLCV_TIMEFRAME)
    period = _yfinance_period(effective_limit, config.OHLCV_TIMEFRAME)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            _log.warning("yfinance returned no data for %s", symbol)
            return None
        hist = hist.tail(effective_limit)
        result: list[tuple[int, float, float, float, float, float]] = []
        for idx, row in hist.iterrows():
            ts = int(idx.timestamp())
            result.append((
                ts,
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Volume"]),
            ))
        _log.debug("yfinance returned %d bars for %s", len(result), symbol)
        return result
    except Exception as exc:
        _log.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def _upsert_candles(
    symbol: str,
    candles: list[tuple[int, float, float, float, float, float]],
) -> int:
    """Upsert stock candles into prices. Returns the number of newly inserted rows."""
    conn = get_connection()
    inserted = 0
    for ts, open_, high, low, close, volume in candles:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO prices
                (symbol, asset_class, timeframe, ts, open, high, low, close, volume)
            VALUES (?, 'stocks', ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, config.OHLCV_TIMEFRAME, ts, open_, high, low, close, volume),
        )
        inserted += cursor.rowcount
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _collect_symbol(symbol: str, limit: int) -> int:
    """Collect OHLCV for one stock symbol. Returns inserted row count (0 on error)."""
    try:
        _log.info("collecting OHLCV for %s (limit=%d)", symbol, limit)
        candles: list[tuple[int, float, float, float, float, float]] | None = None

        bars = _fetch_alpaca_bars(symbol, limit)
        if bars is not None:
            candles = _alpaca_bars_to_candles(bars)

        if not candles:
            _log.info("falling back to yfinance for %s", symbol)
            candles = _fetch_yfinance_bars(symbol, limit)

        if not candles:
            _log.warning("no data collected for %s", symbol)
            return 0

        count = _upsert_candles(symbol, candles)
        _log.info("inserted %d new candles for %s", count, symbol)
        return count

    except Exception as exc:
        _log.error("collection failed for %s: %s", symbol, exc, exc_info=True)
        return 0


def collect_once(
    symbols: list[str] | None = None,
    watchlist: list[str] | None = None,
) -> dict[str, int]:
    """Fetch OHLCV for the universe (pinned ∪ watchlist) and persist to prices.

    Tries Alpaca first for each symbol; falls back to yfinance when Alpaca
    credentials are absent or the call fails.

    Pinned symbols (``symbols``) use the full ``OHLCV_LIMIT`` window.
    Watchlist-only candidates (``watchlist``) use the shorter
    ``WATCHLIST_OHLCV_LIMIT`` window.  Symbols in both lists are treated as
    pinned (full window, collected once).

    Args:
        symbols: Pinned symbols; defaults to config.STOCK_SYMBOLS.
        watchlist: Additional candidate symbols; defaults to config.STOCK_WATCHLIST.

    Returns:
        Mapping of symbol → number of newly inserted candle rows.
    """
    pinned = symbols if symbols is not None else config.STOCK_SYMBOLS
    candidates = watchlist if watchlist is not None else config.STOCK_WATCHLIST

    pinned_set = set(pinned)
    watchlist_only = [s for s in candidates if s not in pinned_set]

    results: dict[str, int] = {}

    for symbol in pinned:
        results[symbol] = _collect_symbol(symbol, config.OHLCV_LIMIT)

    for symbol in watchlist_only:
        results[symbol] = _collect_symbol(symbol, config.WATCHLIST_OHLCV_LIMIT)

    return results


if __name__ == "__main__":
    totals = collect_once()
    for sym, n in totals.items():
        print(f"{sym}: {n} new rows")
