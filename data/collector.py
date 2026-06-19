"""OHLCV + funding rate collector for Binance (via ccxt).

Fetches spot OHLCV candles and the current funding rate from USDT-M
perpetual futures for each crypto symbol, then upserts into the prices
table (INSERT OR IGNORE — unique index on (symbol, timeframe, ts)).
"""

from __future__ import annotations

import time
from typing import Any

import ccxt

import config
from database.db import get_connection, get_logger

_log = get_logger(__name__)

_perp_symbols_cache: set[str] | None = None
_perp_cache_ts: float = 0.0
_PERP_CACHE_TTL = 4 * 3600  # refresh perp symbol list every 4 hours


def _make_exchange(market_type: str = "spot") -> ccxt.binance:
    params: dict[str, Any] = {
        "enableRateLimit": True,
        "options": {"defaultType": market_type},
    }
    if config.BINANCE_API_KEY:
        params["apiKey"] = config.BINANCE_API_KEY
        params["secret"] = config.BINANCE_SECRET
    return ccxt.binance(params)


def _get_perp_symbols(futures: ccxt.binance) -> set[str] | None:
    """Return cached set of spot symbols that have a USDT-M perp on Binance.

    Returns None if the list has never been loaded successfully (caller falls
    back to attempting fetch + warning on failure — original behavior).
    """
    global _perp_symbols_cache, _perp_cache_ts
    now = time.monotonic()
    if _perp_symbols_cache is not None and (now - _perp_cache_ts) < _PERP_CACHE_TTL:
        return _perp_symbols_cache
    try:
        markets = futures.load_markets()
        perps: set[str] = set()
        for market in markets.values():
            if market.get("swap") and market.get("settle") == "USDT":
                base = market.get("base", "")
                if base:
                    perps.add(f"{base}/USDT")
        _perp_symbols_cache = perps
        _perp_cache_ts = now
        _log.info("loaded %d USDT-M perp symbols from Binance", len(perps))
    except Exception as exc:
        _log.warning("could not load Binance perp markets: %s", exc)
        # keep stale cache if available; return None on first-call failure
    return _perp_symbols_cache


def _perp_symbol(spot_symbol: str) -> str:
    """Convert 'BTC/USDT' → 'BTC/USDT:USDT' (Binance USDT-M perpetual)."""
    base = spot_symbol.split("/")[0]
    return f"{base}/USDT:USDT"


def _fetch_funding_rate(
    futures: ccxt.binance,
    spot_symbol: str,
    perp_symbols: set[str] | None = None,
) -> float | None:
    """Return the latest funding rate for the perpetual matching *spot_symbol*.

    When *perp_symbols* is provided and *spot_symbol* is not in it, returns
    None silently (no WARNING) — the coin has no perp listed on Binance.
    """
    if perp_symbols is not None and spot_symbol not in perp_symbols:
        return None
    try:
        result = futures.fetch_funding_rate(_perp_symbol(spot_symbol))
        return result.get("fundingRate")
    except Exception as exc:
        _log.warning("funding_rate unavailable for %s: %s", spot_symbol, exc)
        return None


def _upsert_candles(
    symbol: str,
    candles: list[list[Any]],
    funding_rate: float | None,
) -> int:
    """Insert candles into prices, skipping duplicates. Returns inserted count."""
    conn = get_connection()
    inserted = 0
    for candle in candles:
        ts_ms, open_, high, low, close, volume = candle
        ts = int(ts_ms // 1000)  # ccxt returns milliseconds; schema uses unix seconds
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO prices
                (symbol, asset_class, timeframe, ts, open, high, low, close, volume, funding_rate)
            VALUES (?, 'crypto', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                config.OHLCV_TIMEFRAME,
                ts,
                open_,
                high,
                low,
                close,
                volume,
                funding_rate,
            ),
        )
        inserted += cursor.rowcount
    conn.commit()
    return inserted


def _collect_symbol(
    spot: ccxt.binance,
    futures: ccxt.binance,
    symbol: str,
    limit: int,
    perp_symbols: set[str] | None = None,
) -> int:
    """Collect OHLCV + funding for one symbol. Returns inserted row count (0 on error)."""
    try:
        _log.info("collecting OHLCV for %s (limit=%d)", symbol, limit)
        candles = spot.fetch_ohlcv(symbol, timeframe=config.OHLCV_TIMEFRAME, limit=limit)
        funding_rate = _fetch_funding_rate(futures, symbol, perp_symbols)
        count = _upsert_candles(symbol, candles, funding_rate)
        _log.info(
            "inserted %d new candles for %s (funding_rate=%s)",
            count,
            symbol,
            funding_rate,
        )
        return count
    except Exception as exc:
        _log.error("collection failed for %s: %s", symbol, exc, exc_info=True)
        return 0


def collect_once(
    symbols: list[str] | None = None,
    watchlist: list[str] | None = None,
) -> dict[str, int]:
    """Fetch OHLCV + funding rates for the universe (pinned ∪ watchlist).

    Pinned symbols (``symbols``) are fetched with the full ``OHLCV_LIMIT`` window.
    Watchlist-only candidates (``watchlist``) are fetched with the shorter
    ``WATCHLIST_OHLCV_LIMIT`` window to reduce collection overhead.  Symbols
    present in both lists are treated as pinned (full window, collected once).

    Args:
        symbols: Pinned symbols; defaults to config.CRYPTO_SYMBOLS.
        watchlist: Additional candidate symbols; defaults to config.CRYPTO_WATCHLIST.

    Returns:
        Mapping of symbol → number of newly inserted candle rows.
    """
    pinned = symbols if symbols is not None else config.CRYPTO_SYMBOLS
    candidates = watchlist if watchlist is not None else config.CRYPTO_WATCHLIST

    pinned_set = set(pinned)
    watchlist_only = [s for s in candidates if s not in pinned_set]

    spot = _make_exchange("spot")
    futures = _make_exchange("future")
    perp_symbols = _get_perp_symbols(futures)
    results: dict[str, int] = {}

    for symbol in pinned:
        results[symbol] = _collect_symbol(spot, futures, symbol, config.OHLCV_LIMIT, perp_symbols)

    for symbol in watchlist_only:
        results[symbol] = _collect_symbol(spot, futures, symbol, config.WATCHLIST_OHLCV_LIMIT, perp_symbols)

    return results


if __name__ == "__main__":
    totals = collect_once()
    for sym, n in totals.items():
        print(f"{sym}: {n} new rows")
