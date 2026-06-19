"""Cached set of active USDT spot symbols on Binance (via ccxt load_markets).

Used to filter the universe before collection so BadSymbol errors never occur.
Markets rarely change, so a 1-hour TTL is sufficient.
"""

from __future__ import annotations

import time

import ccxt

from database.db import get_logger

_log = get_logger(__name__)

_CACHE_TTL_SECS: int = 3600

_cache_symbols: frozenset[str] | None = None  # None = never successfully loaded
_cache_ts: float = 0.0


def get_tradeable_symbols() -> frozenset[str] | None:
    """Return active Binance USDT spot symbols in ccxt format (e.g. ``'BTC/USDT'``).

    Cached for one hour.  Returns ``None`` only if markets have never been loaded
    successfully — callers should skip filtering in that case rather than blocking
    the entire universe.
    """
    global _cache_symbols, _cache_ts

    now = time.time()
    if _cache_symbols is not None and (now - _cache_ts) < _CACHE_TTL_SECS:
        return _cache_symbols

    try:
        ex = ccxt.binance({"enableRateLimit": True})
        markets = ex.load_markets()
        symbols: frozenset[str] = frozenset(
            sym
            for sym, m in markets.items()
            if m.get("active") and m.get("quote") == "USDT" and m.get("type") == "spot"
        )
        _cache_symbols = symbols
        _cache_ts = now
        _log.info("Binance spot symbols loaded: %d USDT pairs", len(symbols))
    except Exception as exc:
        _log.warning("Binance load_markets failed — symbol filter may be incomplete: %s", exc)

    return _cache_symbols


def reset_cache() -> None:
    """Clear the in-memory cache (test helper)."""
    global _cache_symbols, _cache_ts
    _cache_symbols = None
    _cache_ts = 0.0
