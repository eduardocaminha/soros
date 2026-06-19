"""Screener — ranks the universe (pinned ∪ watchlist) and selects candidates.

When SCREENER_ENABLED=False (default): returns pinned symbols only, preserving
the existing behaviour.

When SCREENER_ENABLED=True:
  1. Compute 24 h notional volume from the prices table.
  2. Apply SCREENER_MIN_VOLUME_USD liquidity floor.
  3. Fetch the latest composite_score from signals → conviction = |composite_score|.
  4. Use the latest pre-scored sentiment as a gate (reject score < _SENTIMENT_GATE)
     and as a tiebreaker when convictions are equal.
  5. Always include pinned symbols; select at most SCREENER_TOP_N candidates from
     watchlist-only symbols; total candidates per class capped at MAX_OPEN_POSITIONS.

Usage::

    from engine.screener import screen, ScreenerResult
    result = screen()
    # result.selected_crypto — list of crypto symbols to operate this cycle
    # result.selected_stocks — list of stock symbols to operate this cycle
    # result.entries         — full universe with scores (for the dashboard)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from database.db import get_connection, get_logger

_log = get_logger(__name__)

# Minimum sentiment score to pass the gate when SCREENER_ENABLED=True.
# A watchlist symbol with sentiment below this threshold is excluded from
# selection even if its conviction is high.  Neutral (0.0) always passes.
_SENTIMENT_GATE: float = -0.3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScreenerEntry:
    """Score snapshot for one symbol in the universe."""

    symbol: str
    asset_class: str    # 'crypto' | 'stocks'
    is_pinned: bool
    volume_usd_24h: float
    composite_score: float  # latest from signals table, 0.0 when absent
    sentiment_score: float  # latest from sentiment_signals, 0.0 when absent/stale
    conviction: float       # abs(composite_score)
    selected: bool
    reason: str             # 'pinned' | 'screener' | 'volume_floor' | 'sentiment_gate' | 'not_ranked'


@dataclass
class ScreenerResult:
    """Output of one screener pass."""

    selected_crypto: list[str] = field(default_factory=list)
    selected_stocks: list[str] = field(default_factory=list)
    entries: list[ScreenerEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _volume_usd_24h(symbol: str, asset_class: str) -> float:
    """Sum of (close × volume) over the last 24 one-hour candles."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT close, volume FROM prices
        WHERE symbol = ? AND asset_class = ? AND timeframe = ?
        ORDER BY ts DESC LIMIT 24
        """,
        (symbol, asset_class, config.OHLCV_TIMEFRAME),
    ).fetchall()
    if not rows:
        return 0.0
    return sum(float(r["close"]) * float(r["volume"]) for r in rows)


def _latest_composite(symbol: str, asset_class: str) -> float:
    """Latest composite signal score; 0.0 when no signal row exists."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT composite_score FROM signals
        WHERE symbol = ? AND asset_class = ?
        ORDER BY ts DESC LIMIT 1
        """,
        (symbol, asset_class),
    ).fetchone()
    return float(row["composite_score"]) if row else 0.0


def _latest_sentiment(symbol: str, asset_class: str) -> float:
    """Latest sentiment score within SENTIMENT_MAX_AGE_SECONDS; 0.0 when absent/stale."""
    conn = get_connection()
    cutoff = int(time.time()) - config.SENTIMENT_MAX_AGE_SECONDS
    row = conn.execute(
        """
        SELECT score FROM sentiment_signals
        WHERE symbol = ? AND asset_class = ? AND ts >= ?
        ORDER BY ts DESC LIMIT 1
        """,
        (symbol, asset_class, cutoff),
    ).fetchone()
    return float(row["score"]) if row else 0.0


# ---------------------------------------------------------------------------
# Internal: per-class selection
# ---------------------------------------------------------------------------

def _screen_class(
    pinned: list[str],
    watchlist: list[str],
    asset_class: str,
) -> tuple[list[str], list[ScreenerEntry]]:
    """Screen one asset class and return (selected_symbols, entries).

    Pinned symbols are always selected.  Watchlist-only symbols go through
    the volume floor + sentiment gate + conviction ranking.
    """
    entries: list[ScreenerEntry] = []
    pinned_set = set(pinned)
    selected: list[str] = list(pinned)

    for sym in pinned:
        vol = _volume_usd_24h(sym, asset_class)
        comp = _latest_composite(sym, asset_class)
        sent = _latest_sentiment(sym, asset_class)
        entries.append(ScreenerEntry(
            symbol=sym,
            asset_class=asset_class,
            is_pinned=True,
            volume_usd_24h=vol,
            composite_score=comp,
            sentiment_score=sent,
            conviction=abs(comp),
            selected=True,
            reason="pinned",
        ))

    if not config.SCREENER_ENABLED:
        return selected, entries

    candidates: list[ScreenerEntry] = []

    for sym in watchlist:
        if sym in pinned_set:
            continue

        vol = _volume_usd_24h(sym, asset_class)
        comp = _latest_composite(sym, asset_class)
        sent = _latest_sentiment(sym, asset_class)
        conv = abs(comp)

        if vol < config.SCREENER_MIN_VOLUME_USD:
            reason = "volume_floor"
            entry = ScreenerEntry(
                symbol=sym, asset_class=asset_class, is_pinned=False,
                volume_usd_24h=vol, composite_score=comp, sentiment_score=sent,
                conviction=conv, selected=False, reason=reason,
            )
        elif sent < _SENTIMENT_GATE:
            reason = "sentiment_gate"
            entry = ScreenerEntry(
                symbol=sym, asset_class=asset_class, is_pinned=False,
                volume_usd_24h=vol, composite_score=comp, sentiment_score=sent,
                conviction=conv, selected=False, reason=reason,
            )
        else:
            entry = ScreenerEntry(
                symbol=sym, asset_class=asset_class, is_pinned=False,
                volume_usd_24h=vol, composite_score=comp, sentiment_score=sent,
                conviction=conv, selected=False, reason="not_ranked",
            )
            candidates.append(entry)

        entries.append(entry)

    # Rank by conviction DESC, sentiment DESC as tiebreaker
    candidates.sort(key=lambda e: (e.conviction, e.sentiment_score), reverse=True)

    top_n = min(config.SCREENER_TOP_N, config.MAX_OPEN_POSITIONS)
    for entry in candidates[:top_n]:
        entry.selected = True
        entry.reason = "screener"
        selected.append(entry.symbol)

    for entry in candidates[top_n:]:
        entry.reason = "not_ranked"

    _log.info(
        "screener [%s]: pinned=%d candidates=%d selected_watchlist=%d total=%d",
        asset_class,
        len(pinned),
        len(candidates),
        min(len(candidates), top_n),
        len(selected),
    )

    return selected, entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def screen(
    crypto_pinned: list[str] | None = None,
    crypto_watchlist: list[str] | None = None,
    stock_pinned: list[str] | None = None,
    stock_watchlist: list[str] | None = None,
) -> ScreenerResult:
    """Select symbols to operate this cycle.

    Parameters
    ----------
    crypto_pinned:
        Always-operated crypto symbols.  Defaults to ``config.CRYPTO_SYMBOLS``.
    crypto_watchlist:
        Candidate crypto symbols for screener selection.  Defaults to
        ``config.CRYPTO_WATCHLIST``.
    stock_pinned:
        Always-operated stock symbols.  Defaults to ``config.STOCK_SYMBOLS``.
    stock_watchlist:
        Candidate stock symbols for screener selection.  Defaults to
        ``config.STOCK_WATCHLIST``.

    Returns
    -------
    ScreenerResult
        ``selected_crypto`` and ``selected_stocks`` are the symbols to operate.
        ``entries`` contains every universe symbol with its scores (dashboard use).
    """
    if crypto_pinned is None:
        crypto_pinned = list(config.CRYPTO_SYMBOLS)
    if crypto_watchlist is None:
        crypto_watchlist = list(config.CRYPTO_WATCHLIST)
    if stock_pinned is None:
        stock_pinned = list(config.STOCK_SYMBOLS)
    if stock_watchlist is None:
        stock_watchlist = list(config.STOCK_WATCHLIST)

    sel_crypto, entries_crypto = _screen_class(crypto_pinned, crypto_watchlist, "crypto")
    sel_stocks, entries_stocks = _screen_class(stock_pinned, stock_watchlist, "stocks")

    return ScreenerResult(
        selected_crypto=sel_crypto,
        selected_stocks=sel_stocks,
        entries=entries_crypto + entries_stocks,
    )


def save_screener_result(result: ScreenerResult) -> None:
    """Persist screener entries to screener_runs for dashboard display."""
    if not result.entries:
        return
    conn = get_connection()
    run_ts = int(time.time())
    conn.executemany(
        """
        INSERT INTO screener_runs
            (run_ts, symbol, asset_class, is_pinned, volume_usd_24h,
             composite_score, sentiment_score, conviction, selected, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_ts,
                e.symbol,
                e.asset_class,
                1 if e.is_pinned else 0,
                e.volume_usd_24h,
                e.composite_score,
                e.sentiment_score,
                e.conviction,
                1 if e.selected else 0,
                e.reason,
            )
            for e in result.entries
        ],
    )
    conn.commit()


if __name__ == "__main__":
    result = screen()
    print(f"crypto: {result.selected_crypto}")
    print(f"stocks: {result.selected_stocks}")
    print()
    for e in result.entries:
        tag = "[pinned]" if e.is_pinned else "[watch]"
        sel = "✓" if e.selected else "✗"
        print(
            f"{sel} {tag} {e.symbol:15s}  "
            f"vol_usd={e.volume_usd_24h:>14,.0f}  "
            f"composite={e.composite_score:+.3f}  "
            f"sent={e.sentiment_score:+.3f}  "
            f"({e.reason})"
        )
