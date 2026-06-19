"""Market-cap base tier — top-N crypto by market cap via CoinGecko (keyless).

Fetches /coins/markets ordered by market_cap_desc, converts symbols to
ccxt-compatible USDT pairs, and caches results for MARKETCAP_REFRESH_SECS.
Gracefully degrades to the cached list (or empty) when the API is unavailable.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

import config
from database.db import get_logger

_log = get_logger(__name__)

# Tokens that should never appear in the trading universe.
# Stablecoins have no independent price discovery; wrapped assets and
# liquid-staking tokens shadow their underlying (BTC, ETH) and cause
# duplicate exposure.
_EXCLUDE: frozenset[str] = frozenset({
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX",
    "LUSD", "SUSD", "USDD", "FDUSD", "PYUSD", "USDS",
    "WBTC", "WETH",
    "STETH", "WSTETH",
})

_COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page={per_page}&page=1&sparkline=false"
)

_cache_symbols: list[str] = []
_cache_ts: float = 0.0


def _fetch_from_api(n: int) -> list[str]:
    """Call CoinGecko and return up to *n* ccxt USDT-pair symbols."""
    # Fetch extra rows to cover filtered-out stablecoins/wrapped assets.
    per_page = min(n * 3, 250)
    url = _COINGECKO_URL.format(per_page=per_page)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data: list[dict[str, Any]] = json.loads(resp.read())

    symbols: list[str] = []
    for coin in data:
        ticker = (coin.get("symbol") or "").upper()
        if ticker in _EXCLUDE:
            continue
        symbols.append(f"{ticker}/USDT")
        if len(symbols) >= n:
            break
    return symbols


def get_base_universe(
    n: int | None = None,
    refresh_secs: int | None = None,
) -> list[str]:
    """Return top-*n* crypto symbols by market cap as ccxt ``TICKER/USDT`` pairs.

    Results are cached for *refresh_secs* seconds (default: ``MARKETCAP_REFRESH_SECS``).
    Falls back to the cached list — or an empty list if no cache exists — when
    CoinGecko is unreachable.
    """
    global _cache_symbols, _cache_ts

    if n is None:
        n = config.MARKETCAP_TOP_N
    if refresh_secs is None:
        refresh_secs = config.MARKETCAP_REFRESH_SECS

    now = time.time()
    if _cache_symbols and (now - _cache_ts) < refresh_secs:
        _log.debug(
            "market cap cache hit: %d symbols, age=%.0fs",
            len(_cache_symbols),
            now - _cache_ts,
        )
        return list(_cache_symbols)

    try:
        symbols = _fetch_from_api(n)
        _cache_symbols = symbols
        _cache_ts = now
        _log.info("market cap base updated: %d symbols (top %d)", len(symbols), n)
        return list(symbols)
    except Exception as exc:
        _log.warning(
            "CoinGecko fetch failed — using cached list (%d symbols): %s",
            len(_cache_symbols),
            exc,
        )
        return list(_cache_symbols)


def reset_cache() -> None:
    """Clear the in-memory cache (test helper)."""
    global _cache_symbols, _cache_ts
    _cache_symbols = []
    _cache_ts = 0.0
