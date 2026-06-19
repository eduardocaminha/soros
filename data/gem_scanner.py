"""CEX ignition scanner — spot volume surges via ccxt.fetch_tickers.

Makes a single keyless call to Binance spot fetch_tickers, then:
1. Filters to /USDT pairs only.
2. Removes stablecoins, leveraged tokens, and duplicates from the base tier.
3. Detects volume surges (quoteVolume >= GEM_VOLUME_SURGE_MULTIPLIER × rolling avg).
4. Applies GEM_ROC_MIN_PCT and GEM_MIN_VOLUME_USD floors.
5. Ranks by gem_score (surge_ratio × roc_pct) and returns top GEM_TOP_N candidates.

Rolling average is maintained in-memory across successive scans.  On the first
scan after startup the history is empty, so no gems are surfaced (cold start).
Subsequent calls accumulate a window of up to _VOLUME_HISTORY_MAXLEN snapshots.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import ccxt

import config
from database.db import get_logger

_log = get_logger(__name__)

# Tokens that should never appear as gem candidates.
_STABLE_EXCLUDE: frozenset[str] = frozenset({
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX",
    "LUSD", "SUSD", "USDD", "FDUSD", "PYUSD", "USDS",
    "WBTC", "WETH",
    "STETH", "WSTETH",
})

# Binance leveraged-token suffixes: UP/DOWN/BULL/BEAR and numeric multipliers.
# Examples: BTCUP, ETHDOWN, ETHBULL, BTCBEAR, BTC3L, ETH3S, BNB2L
_LEVERAGED_RE = re.compile(r"(UP|DOWN|BULL|BEAR|\d+L|\d+S)$", re.IGNORECASE)

# Rolling 24 h volume history per symbol (last N snapshots).
_VOLUME_HISTORY_MAXLEN: int = 5
_volume_history: dict[str, deque[float]] = {}


@dataclass
class GemCandidate:
    """Ignition candidate emitted by the CEX scanner."""

    symbol: str               # ccxt format, e.g. 'XYZ/USDT'
    volume_usd_24h: float     # 24 h notional USDT volume at scan time
    volume_surge_ratio: float # current_volume / rolling_avg_volume
    roc_pct: float            # 24 h price change (%)
    gem_score: float          # surge_ratio × roc_pct (ranking key)


def _is_excluded(base_token: str) -> bool:
    """True when this base token is a stable/wrapped/leveraged asset."""
    return base_token in _STABLE_EXCLUDE or bool(_LEVERAGED_RE.search(base_token))


def _make_exchange() -> ccxt.binance:
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def _record_volume(symbol: str, volume: float) -> float | None:
    """Append *volume* to the rolling history for *symbol*.

    Returns the average of the *previous* entries (before this call), or
    ``None`` when this is the first data point (no baseline yet).
    """
    hist = _volume_history.get(symbol)
    if hist is None:
        _volume_history[symbol] = deque([volume], maxlen=_VOLUME_HISTORY_MAXLEN)
        return None
    avg = sum(hist) / len(hist)
    hist.append(volume)
    return avg


def scan_gems(
    exchange: Any | None = None,
    base_symbols: set[str] | None = None,
    top_n: int | None = None,
) -> list[GemCandidate]:
    """Scan Binance spot for ignition candidates.

    Parameters
    ----------
    exchange:
        ccxt exchange instance.  Defaults to a keyless Binance spot instance.
    base_symbols:
        Symbols already in the market-cap base tier (ccxt format, e.g.
        ``{'BTC/USDT', 'ETH/USDT'}``).  These are excluded from gem candidates
        to prevent duplicates in the assembled universe.
    top_n:
        Maximum number of candidates to return.  Defaults to ``config.GEM_TOP_N``.

    Returns
    -------
    list[GemCandidate]
        Candidates ranked by gem_score DESC, length ≤ *top_n*.
        Returns ``[]`` on the first call (cold start, no rolling baseline yet)
        or when the exchange call fails.
    """
    if top_n is None:
        top_n = config.GEM_TOP_N
    if exchange is None:
        exchange = _make_exchange()
    base_set: set[str] = base_symbols or set()

    try:
        tickers: dict[str, Any] = exchange.fetch_tickers()
    except Exception as exc:
        _log.warning("gem_scanner: fetch_tickers failed — %s", exc)
        return []

    candidates: list[GemCandidate] = []
    screened = 0

    for symbol, ticker in tickers.items():
        if not symbol.endswith("/USDT"):
            continue

        base_token = symbol.split("/")[0]
        if _is_excluded(base_token):
            continue
        if symbol in base_set:
            continue

        screened += 1
        volume_usd = float(ticker.get("quoteVolume") or 0.0)
        roc_pct = float(ticker.get("percentage") or 0.0)

        # Always update history so the baseline improves over successive scans,
        # even when the symbol fails one of the filters below.
        rolling_avg = _record_volume(symbol, volume_usd)

        if volume_usd < config.GEM_MIN_VOLUME_USD:
            continue
        if roc_pct < config.GEM_ROC_MIN_PCT:
            continue
        if rolling_avg is None:
            # Cold start: no baseline yet for this symbol.
            continue
        if rolling_avg <= 0:
            continue

        surge_ratio = volume_usd / rolling_avg
        if surge_ratio < config.GEM_VOLUME_SURGE_MULTIPLIER:
            continue

        gem_score = surge_ratio * roc_pct
        candidates.append(GemCandidate(
            symbol=symbol,
            volume_usd_24h=volume_usd,
            volume_surge_ratio=surge_ratio,
            roc_pct=roc_pct,
            gem_score=gem_score,
        ))

    candidates.sort(key=lambda c: c.gem_score, reverse=True)
    result = candidates[:top_n]

    _log.info(
        "gem_scanner: screened=%d surges=%d gems=%d",
        screened,
        len(candidates),
        len(result),
    )
    return result


def reset_history() -> None:
    """Clear the in-memory volume history (test helper)."""
    _volume_history.clear()
