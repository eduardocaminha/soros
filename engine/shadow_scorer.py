"""Forward shadow scoring: compute real + keyless-only composites each live cycle.

At each live cycle, this module:
  1. Computes a shadow (keyless-only) sentiment score per symbol using only
     Fear & Greed (alternative.me) and CoinGecko community votes — never calls
     Claude or any subscription service.
  2. Produces two composite variants per symbol:
     - real: composite_score from the real aggregation (whatever config says)
     - shadow: composite recomputed with keyless-only sentiment
  3. Runs a virtual paper simulation for each variant, updating
     forward_shadow_positions in SQLite.
  4. Snapshots virtual equity per variant to forward_shadow_equity each cycle.

Callers MUST wrap this module's entry point in try/except — shadow failures
must never propagate to the real trading cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
from database.db import get_connection, get_logger
from engine.signal_aggregator import AggregatedSignal, _action, _final_composite
from sentiment.sources_crypto import _fetch_coingecko_sentiment, _fetch_fear_greed

_log = get_logger(__name__)

VARIANT_REAL = "real"
VARIANT_SHADOW = "shadow"


@dataclass
class ShadowScore:
    """Per-symbol shadow scoring result: real variant + shadow (keyless) variant."""

    symbol: str
    asset_class: str
    # Real variant — composite from the live aggregation (may include Claude)
    real_composite: float
    real_action: str
    # Shadow variant — composite using only F&G + CoinGecko votes
    keyless_sentiment: float
    shadow_composite: float
    shadow_action: str
    # Latest market price used for mark-to-market (None when unavailable)
    current_price: float | None


# ---------------------------------------------------------------------------
# Keyless sentiment
# ---------------------------------------------------------------------------

def compute_keyless_sentiment(
    symbol: str,
    asset_class: str,
    *,
    fg_value: int | None = None,
) -> float:
    """Return keyless sentiment score for *symbol* in [-1, 1].

    Uses only Fear & Greed (alternative.me) and CoinGecko community votes.
    Never calls Claude or any subscription service.

    For stocks, always returns 0.0 (no keyless sentiment source available).

    Args:
        symbol:     Trading symbol, e.g. ``'BTC/USDT'``.
        asset_class: ``'crypto'`` or ``'stocks'``.
        fg_value:   Pre-fetched F&G value (0–100) to avoid duplicate HTTP calls
                    when processing multiple symbols in the same cycle. When
                    None, this function fetches it from alternative.me.
    """
    if asset_class != "crypto":
        return 0.0

    scores: list[float] = []

    # Fear & Greed Index
    if fg_value is None:
        fg_raw, _ = _fetch_fear_greed()
    else:
        fg_raw = fg_value
    if fg_raw is not None:
        scores.append(max(-1.0, min(1.0, (fg_raw - 50) / 50.0)))

    # CoinGecko community sentiment votes
    base = symbol.split("/")[0].upper()
    cg_sent = _fetch_coingecko_sentiment(base)
    if cg_sent is not None:
        scores.append(cg_sent)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Shadow composite computation
# ---------------------------------------------------------------------------

def compute_shadow_scores(aggregated: list[AggregatedSignal]) -> list[ShadowScore]:
    """Compute real + shadow composites for every aggregated signal.

    Fetches the Fear & Greed index once per call (shared across all crypto
    symbols) to avoid redundant HTTP requests.

    Args:
        aggregated: Output of ``signal_aggregator.aggregate_once()``, already
                    written to the DB.  Each entry provides the deterministic
                    sub-scores and the real composite.

    Returns:
        One ``ShadowScore`` per signal that could be processed.  Symbols that
        fail (e.g. price lookup error) are logged at WARNING and skipped.
    """
    # Fetch F&G once for all crypto symbols
    fg_raw: int | None = None
    has_crypto = any(s.asset_class == "crypto" for s in aggregated)
    if has_crypto:
        fg_raw, _ = _fetch_fear_greed()

    conn = get_connection()
    results: list[ShadowScore] = []

    for sig in aggregated:
        try:
            sent = compute_keyless_sentiment(sig.symbol, sig.asset_class, fg_value=fg_raw)
            shadow_comp = _final_composite(
                sig.momentum_score,
                sig.volatility_score,
                sig.funding_score,
                sent,
                sig.asset_class,
                ign=sig.ignition_score,
            )
            shadow_act = _action(shadow_comp)

            price_row = conn.execute(
                "SELECT close FROM prices WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
                (sig.symbol,),
            ).fetchone()
            price = float(price_row["close"]) if price_row else None

            results.append(
                ShadowScore(
                    symbol=sig.symbol,
                    asset_class=sig.asset_class,
                    real_composite=sig.composite_score,
                    real_action=sig.action,
                    keyless_sentiment=sent,
                    shadow_composite=shadow_comp,
                    shadow_action=shadow_act,
                    current_price=price,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("shadow score failed for %s: %s", sig.symbol, exc)

    return results


# ---------------------------------------------------------------------------
# Virtual position / equity helpers
# ---------------------------------------------------------------------------

def _virtual_equity(variant: str) -> float:
    """Compute virtual equity for *variant* from forward_shadow_positions.

    Equity = INITIAL_CAPITAL + realized P&L + unrealized P&L.
    Unrealized P&L is computed from the latest price in the prices table.
    """
    conn = get_connection()

    realized = float(
        conn.execute(
            """
            SELECT COALESCE(SUM((exit_price - entry_price) * quantity), 0.0)
            FROM forward_shadow_positions
            WHERE variant = ? AND status = 'closed'
            """,
            (variant,),
        ).fetchone()[0]
    )

    open_rows = conn.execute(
        """
        SELECT symbol, quantity, entry_price
        FROM forward_shadow_positions
        WHERE variant = ? AND status = 'open'
        """,
        (variant,),
    ).fetchall()

    unrealized = 0.0
    for row in open_rows:
        price_row = conn.execute(
            "SELECT close FROM prices WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            (row["symbol"],),
        ).fetchone()
        if price_row is not None:
            unrealized += (float(price_row["close"]) - float(row["entry_price"])) * float(row["quantity"])

    return config.INITIAL_CAPITAL + realized + unrealized


def _open_position(
    variant: str,
    symbol: str,
    asset_class: str,
    entry_price: float,
    equity: float,
) -> None:
    """Open a virtual long position for *variant* if none is open."""
    conn = get_connection()
    existing = conn.execute(
        """
        SELECT id FROM forward_shadow_positions
        WHERE variant = ? AND symbol = ? AND status = 'open'
        """,
        (variant, symbol),
    ).fetchone()
    if existing is not None:
        return  # already holding

    allocation = equity * config.POSITION_SIZE_PCT
    if allocation <= 0.0 or entry_price <= 0.0:
        return
    quantity = allocation / entry_price

    conn.execute(
        """
        INSERT INTO forward_shadow_positions
            (variant, symbol, asset_class, side, status, quantity, entry_price)
        VALUES (?, ?, ?, 'long', 'open', ?, ?)
        """,
        (variant, symbol, asset_class, quantity, entry_price),
    )
    conn.commit()
    _log.debug(
        "shadow open [%s] %s qty=%.6f price=%.4f",
        variant, symbol, quantity, entry_price,
    )


def _close_position(
    variant: str,
    symbol: str,
    exit_price: float,
) -> None:
    """Close the open virtual long position for *variant* / *symbol*."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT id FROM forward_shadow_positions
        WHERE variant = ? AND symbol = ? AND status = 'open'
        """,
        (variant, symbol),
    ).fetchone()
    if row is None:
        return  # nothing to close

    conn.execute(
        """
        UPDATE forward_shadow_positions
        SET status = 'closed', exit_price = ?, closed_at = ?
        WHERE id = ?
        """,
        (exit_price, int(time.time()), int(row["id"])),
    )
    conn.commit()
    _log.debug("shadow close [%s] %s price=%.4f", variant, symbol, exit_price)


def _record_equity_snapshot(
    variant: str,
    equity: float,
    is_paper: bool,
) -> None:
    """Snapshot virtual equity for *variant* into forward_shadow_equity."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT peak_equity FROM forward_shadow_equity
        WHERE variant = ?
        ORDER BY ts DESC, id DESC LIMIT 1
        """,
        (variant,),
    ).fetchone()

    peak = float(row["peak_equity"]) if row else equity
    if equity > peak:
        peak = equity
    drawdown = (peak - equity) / peak if peak > 0.0 else 0.0

    conn.execute(
        """
        INSERT INTO forward_shadow_equity
            (variant, equity, peak_equity, drawdown_pct, is_paper)
        VALUES (?, ?, ?, ?, ?)
        """,
        (variant, equity, peak, drawdown, int(is_paper)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tick(
    aggregated: list[AggregatedSignal],
    *,
    is_paper: bool = True,
) -> list[ShadowScore]:
    """Run one shadow scoring cycle.

    Computes shadow scores, updates virtual positions for both variants,
    and snapshots virtual equity. Must be called AFTER the real aggregation
    so ``AggregatedSignal.composite_score`` values are authoritative.

    Args:
        aggregated: Real aggregation results from this cycle.
        is_paper:   Passed through to the equity snapshot (mirrors real bot).

    Returns:
        List of ``ShadowScore`` for logging / debug; callers may discard it.

    Raises:
        Never — all errors are caught and logged internally.
    """
    scores = compute_shadow_scores(aggregated)

    for score in scores:
        if score.current_price is None or score.current_price <= 0.0:
            _log.debug("shadow tick: no price for %s, skipping position update", score.symbol)
            continue

        for variant, action in (
            (VARIANT_REAL, score.real_action),
            (VARIANT_SHADOW, score.shadow_action),
        ):
            try:
                if action == "buy":
                    equity = _virtual_equity(variant)
                    _open_position(
                        variant,
                        score.symbol,
                        score.asset_class,
                        score.current_price,
                        equity,
                    )
                elif action == "sell":
                    _close_position(variant, score.symbol, score.current_price)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "shadow position update failed [%s] %s: %s",
                    variant, score.symbol, exc,
                )

    for variant in (VARIANT_REAL, VARIANT_SHADOW):
        try:
            equity = _virtual_equity(variant)
            _record_equity_snapshot(variant, equity, is_paper)
            _log.info(
                "shadow equity [%s]: %.2f (paper=%s)",
                variant, equity, is_paper,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("shadow equity snapshot failed [%s]: %s", variant, exc)

    return scores
