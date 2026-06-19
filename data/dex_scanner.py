"""DEX discovery signals — trending tokens from DexScreener and GeckoTerminal.

Both sources are keyless. Results are used exclusively as a gem_score boost for
tokens that also have a Binance spot pair (CEX-only execution).

Flow:
- DexScreener: /token-boosts/top/v1 to get top boosted token addresses, then
  /latest/dex/tokens/{addresses} (batch, comma-separated) to resolve symbols.
- GeckoTerminal: /api/v2/networks/trending_pools?include=base_token to get
  trending pool data with base token symbols via JSON:API includes.

Results are cached for DEX_SCAN_CACHE_SECS seconds.  Both sources degrade
gracefully: a failed source contributes zero symbols; the other's results still
apply.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

import config
from database.db import get_logger

_log = get_logger(__name__)

_DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/top/v1"
_DEXSCREENER_TOKENS_URL = "https://api.dexscreener.com/latest/dex/tokens/{addresses}"
_GECKOTERMINAL_TRENDING_URL = (
    "https://api.geckoterminal.com/api/v2/networks/trending_pools"
    "?include=base_token&page=1"
)

# DexScreener returns up to 30 addresses per batch lookup.
_DEX_BATCH_SIZE = 20

_cache_symbols: frozenset[str] = frozenset()
_cache_ts: float = 0.0


def _http_get(url: str, timeout: int = 10) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (soros-bot)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _fetch_dexscreener() -> set[str]:
    """Return base token symbols from DexScreener top boosted tokens.

    Two-step: fetch boost list for addresses, then batch-resolve symbols.
    """
    boosts = _http_get(_DEXSCREENER_BOOSTS_URL)
    if not isinstance(boosts, list) or not boosts:
        return set()

    seen: set[str] = set()
    addresses: list[str] = []
    for item in boosts[:_DEX_BATCH_SIZE]:
        addr = (item.get("tokenAddress") or "").strip()
        if addr and addr not in seen:
            seen.add(addr)
            addresses.append(addr)

    if not addresses:
        return set()

    data = _http_get(_DEXSCREENER_TOKENS_URL.format(addresses=",".join(addresses)))
    pairs = data.get("pairs") or []

    symbols: set[str] = set()
    for pair in pairs:
        symbol = ((pair.get("baseToken") or {}).get("symbol") or "").strip()
        if symbol:
            symbols.add(symbol.upper())
    return symbols


def _fetch_geckoterminal() -> set[str]:
    """Return base token symbols from GeckoTerminal trending pools.

    Uses JSON:API ``included`` records (type=token) to resolve symbols.
    Falls back to parsing the pool ``name`` attribute ("TOKEN / QUOTE ...").
    """
    data = _http_get(_GECKOTERMINAL_TRENDING_URL)

    # Build id → symbol lookup from included tokens.
    token_by_id: dict[str, str] = {}
    for item in data.get("included") or []:
        if item.get("type") == "token":
            token_id = item.get("id") or ""
            symbol = ((item.get("attributes") or {}).get("symbol") or "").strip()
            if token_id and symbol:
                token_by_id[token_id] = symbol.upper()

    symbols: set[str] = set()
    for pool in data.get("data") or []:
        # Primary: resolve via relationships → included token.
        base_ref = (pool.get("relationships") or {}).get("base_token", {})
        token_id = (base_ref.get("data") or {}).get("id") or ""
        if token_id and token_id in token_by_id:
            symbols.add(token_by_id[token_id])
            continue

        # Fallback: parse pool name like "PEPE / WETH 1%".
        name = ((pool.get("attributes") or {}).get("name") or "").strip()
        if " / " in name:
            base = name.split(" / ")[0].strip()
            if base:
                symbols.add(base.upper())

    return symbols


def get_dex_trending_symbols(cache_secs: int | None = None) -> frozenset[str]:
    """Return base token symbols currently trending on DEX platforms.

    Merges results from DexScreener and GeckoTerminal. Results are cached for
    *cache_secs* seconds (default: ``config.DEX_SCAN_CACHE_SECS``). Caches an
    empty set on total failure to rate-limit retries within the same window.
    """
    global _cache_symbols, _cache_ts

    if cache_secs is None:
        cache_secs = config.DEX_SCAN_CACHE_SECS

    now = time.time()
    if _cache_ts > 0 and (now - _cache_ts) < cache_secs:
        _log.debug("dex_scanner: cache hit (%d symbols)", len(_cache_symbols))
        return _cache_symbols

    combined: set[str] = set()
    for name, fn in [
        ("DexScreener", _fetch_dexscreener),
        ("GeckoTerminal", _fetch_geckoterminal),
    ]:
        try:
            result = fn()
            combined |= result
            _log.debug("dex_scanner: %s returned %d symbols", name, len(result))
        except Exception as exc:
            _log.warning("dex_scanner: %s fetch failed — %s", name, exc)

    _cache_symbols = frozenset(combined)
    _cache_ts = now
    _log.info("dex_scanner: cached %d DEX trending symbols", len(_cache_symbols))
    return _cache_symbols


def reset_cache() -> None:
    """Clear cached symbols (test helper)."""
    global _cache_symbols, _cache_ts
    _cache_symbols = frozenset()
    _cache_ts = 0.0
