"""OHLCV + funding rate collector for Binance (via ccxt).

Fetches spot OHLCV candles and the current funding rate from USDT-M
perpetual futures for each crypto symbol, then upserts into the prices
table (INSERT OR IGNORE — unique index on (symbol, timeframe, ts)).
"""

from __future__ import annotations

from typing import Any

import ccxt

import config
from database.db import get_connection, get_logger

_log = get_logger(__name__)


def _make_exchange(market_type: str = "spot") -> ccxt.binance:
    params: dict[str, Any] = {
        "enableRateLimit": True,
        "options": {"defaultType": market_type},
    }
    if config.BINANCE_API_KEY:
        params["apiKey"] = config.BINANCE_API_KEY
        params["secret"] = config.BINANCE_SECRET
    return ccxt.binance(params)


def _perp_symbol(spot_symbol: str) -> str:
    """Convert 'BTC/USDT' → 'BTC/USDT:USDT' (Binance USDT-M perpetual)."""
    base = spot_symbol.split("/")[0]
    return f"{base}/USDT:USDT"


def _fetch_funding_rate(
    futures: ccxt.binance, spot_symbol: str
) -> float | None:
    """Return the latest funding rate for the perpetual matching *spot_symbol*."""
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


def collect_once(symbols: list[str] | None = None) -> dict[str, int]:
    """Fetch OHLCV + funding rates for each crypto symbol and persist to prices.

    Args:
        symbols: override the symbol list; defaults to config.CRYPTO_SYMBOLS.

    Returns:
        Mapping of symbol → number of newly inserted candle rows.
    """
    symbols = symbols or config.CRYPTO_SYMBOLS
    spot = _make_exchange("spot")
    futures = _make_exchange("future")
    results: dict[str, int] = {}

    for symbol in symbols:
        try:
            _log.info("collecting OHLCV for %s", symbol)
            candles = spot.fetch_ohlcv(
                symbol,
                timeframe=config.OHLCV_TIMEFRAME,
                limit=config.OHLCV_LIMIT,
            )
            funding_rate = _fetch_funding_rate(futures, symbol)
            count = _upsert_candles(symbol, candles, funding_rate)
            _log.info(
                "inserted %d new candles for %s (funding_rate=%s)",
                count,
                symbol,
                funding_rate,
            )
            results[symbol] = count
        except Exception as exc:
            _log.error("collection failed for %s: %s", symbol, exc, exc_info=True)
            results[symbol] = 0

    return results


if __name__ == "__main__":
    totals = collect_once()
    for sym, n in totals.items():
        print(f"{sym}: {n} new rows")
