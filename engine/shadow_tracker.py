"""Forward shadow scoring — keyless sentiment A/B equity tracker.

Each live cycle, snapshots two virtual equity tracks:
  - real:   current mark-to-market equity of the configured bot, tagged 'real'.
  - shadow: virtual equity accumulated by a paper book driven by the keyless
            composite (Fear&Greed + CoinGecko votes for crypto; CNN F&G +
            Yahoo Finance for stocks).  NEVER calls Claude/subscription —
            no quota is consumed by the shadow simulation.

Entry point: snapshot_forward_ab(real_equity, aggregated)

All shadow computation is best-effort: any exception is logged and swallowed
so the caller's real trade cycle is never interrupted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
from database.db import get_connection, get_logger
from engine.signal_aggregator import AggregatedSignal, _action, _final_composite
from sentiment import sources_crypto, sources_stocks

_log = get_logger(__name__)


@dataclass
class ShadowScore:
    """Keyless composite result for one symbol."""

    symbol: str
    asset_class: str
    keyless_sentiment: float
    shadow_composite: float
    shadow_action: str  # 'buy' | 'sell' | 'hold'


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_shadow_scores(aggregated: list[AggregatedSignal]) -> list[ShadowScore]:
    """Compute keyless-sentiment composite for each aggregated signal.

    Fetches Fear&Greed + CoinGecko votes (crypto) or CNN F&G + Yahoo Finance
    (stocks) — no LLM / Claude subscription.  Deterministic sub-scores are
    taken from the already-aggregated signals; only the sentiment component
    is replaced with the keyless value.

    Fetch failures for individual symbols are logged and skipped; a partial
    list is returned rather than raising.
    """
    results: list[ShadowScore] = []
    for sig in aggregated:
        try:
            if sig.asset_class == "crypto":
                src = sources_crypto.fetch(sig.symbol)
                keyless_sent = sources_crypto.pre_score(src)
            else:
                # Pass empty key so Finnhub (keyed) is skipped — truly keyless.
                src = sources_stocks.fetch(sig.symbol, finnhub_api_key="")
                keyless_sent = sources_stocks.pre_score(src)

            shadow_comp = _final_composite(
                sig.momentum_score,
                sig.volatility_score,
                sig.funding_score,
                keyless_sent,
                sig.asset_class,
                ign=sig.ignition_score,
            )
            results.append(ShadowScore(
                symbol=sig.symbol,
                asset_class=sig.asset_class,
                keyless_sentiment=keyless_sent,
                shadow_composite=shadow_comp,
                shadow_action=_action(shadow_comp),
            ))
        except Exception as exc:  # noqa: BLE001
            _log.warning("shadow score failed for %s: %s", sig.symbol, exc)
    return results


# ---------------------------------------------------------------------------
# Virtual paper book
# ---------------------------------------------------------------------------


def _latest_close(symbol: str) -> float | None:
    """Most recent close price for *symbol* from the prices table."""
    conn = get_connection()
    row = conn.execute(
        "SELECT close FROM prices WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return float(row["close"]) if row else None


def _shadow_equity() -> float:
    """Virtual shadow equity: INITIAL_CAPITAL + realized P&L + unrealized P&L."""
    conn = get_connection()
    realized = float(conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0.0) AS t"
        " FROM shadow_positions WHERE status = 'closed'"
    ).fetchone()["t"])
    unrealized = float(conn.execute(
        "SELECT COALESCE(SUM(unrealized_pnl), 0.0) AS t"
        " FROM shadow_positions WHERE status = 'open'"
    ).fetchone()["t"])
    return config.INITIAL_CAPITAL + realized + unrealized


def _open_shadow_count() -> int:
    conn = get_connection()
    return int(conn.execute(
        "SELECT COUNT(*) FROM shadow_positions WHERE status = 'open'"
    ).fetchone()[0])


def _update_shadow_book(scores: list[ShadowScore]) -> None:
    """Update virtual shadow positions based on shadow composite actions.

    Mirrors the real paper executor logic:
    - 'buy'  + no open position → open long (if under MAX_OPEN_POSITIONS)
    - 'sell' + open position    → close long, realise P&L
    - 'hold'                    → no change
    After all action updates, marks every open position to market.
    """
    conn = get_connection()
    current_equity = _shadow_equity()

    for score in scores:
        price = _latest_close(score.symbol)
        if price is None:
            continue

        open_pos = conn.execute(
            "SELECT id, quantity, entry_price FROM shadow_positions"
            " WHERE symbol = ? AND status = 'open'",
            (score.symbol,),
        ).fetchone()

        if score.shadow_action == "buy" and open_pos is None:
            if _open_shadow_count() < config.MAX_OPEN_POSITIONS:
                effective_price = price * (1 + config.FEE_PCT + config.SLIPPAGE_PCT)
                if effective_price > 0:
                    size = current_equity * config.POSITION_SIZE_PCT
                    qty = size / effective_price
                    conn.execute(
                        """
                        INSERT INTO shadow_positions
                            (symbol, asset_class, side, quantity, entry_price,
                             current_price, unrealized_pnl)
                        VALUES (?, ?, 'long', ?, ?, ?, 0.0)
                        """,
                        (score.symbol, score.asset_class, qty, effective_price, price),
                    )
                    _log.info(
                        "shadow open long: %s qty=%.6f price=%.4f",
                        score.symbol, qty, effective_price,
                    )

        elif score.shadow_action == "sell" and open_pos is not None:
            qty = float(open_pos["quantity"])
            entry = float(open_pos["entry_price"])
            effective_exit = price * (1 - config.FEE_PCT - config.SLIPPAGE_PCT)
            pnl = (effective_exit - entry) * qty
            conn.execute(
                """
                UPDATE shadow_positions
                SET status = 'closed', closed_at = ?, current_price = ?,
                    unrealized_pnl = 0.0, realized_pnl = ?
                WHERE id = ?
                """,
                (int(time.time()), price, pnl, int(open_pos["id"])),
            )
            _log.info("shadow close: %s pnl=%.4f", score.symbol, pnl)

    # Mark-to-market: refresh unrealized P&L for all open shadow positions.
    open_rows = conn.execute(
        "SELECT id, symbol, quantity, entry_price FROM shadow_positions WHERE status = 'open'"
    ).fetchall()
    for row in open_rows:
        price = _latest_close(row["symbol"])
        if price is None:
            continue
        unrealized = (price - float(row["entry_price"])) * float(row["quantity"])
        conn.execute(
            "UPDATE shadow_positions SET current_price = ?, unrealized_pnl = ? WHERE id = ?",
            (price, unrealized, int(row["id"])),
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _record_snapshot(variant: str, equity: float, *, is_paper: bool = True) -> None:
    """Persist one equity snapshot tagged by *variant* ('real' | 'shadow')."""
    conn = get_connection()
    row = conn.execute(
        "SELECT peak_equity FROM forward_shadow_snapshots"
        " WHERE variant = ? ORDER BY ts DESC, id DESC LIMIT 1",
        (variant,),
    ).fetchone()

    peak = row["peak_equity"] if row else equity
    if equity > peak:
        peak = equity
    drawdown = (peak - equity) / peak if peak > 0.0 else 0.0

    conn.execute(
        """
        INSERT INTO forward_shadow_snapshots (variant, equity, peak_equity, drawdown_pct, is_paper)
        VALUES (?, ?, ?, ?, ?)
        """,
        (variant, equity, peak, drawdown, int(is_paper)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def snapshot_forward_ab(
    real_equity: float,
    aggregated: list[AggregatedSignal],
    *,
    is_paper: bool = True,
) -> None:
    """Compute shadow scores, update virtual book, and snapshot both equity tracks.

    Called once per live cycle from main.py.  Snapshots:
      - 'real'   variant using *real_equity* (current bot equity).
      - 'shadow' variant using the virtual equity driven by keyless sentiment.

    All shadow computation is best-effort; any exception is logged and swallowed
    so the caller's trade cycle is never interrupted.

    Args:
        real_equity: Current mark-to-market equity of the configured bot.
        aggregated:  Signals aggregated this cycle; provides deterministic
                     sub-scores — shadow replaces only the sentiment component.
        is_paper:    Forwarded to both snapshot rows (mirrors real bot's mode).
    """
    try:
        _record_snapshot("real", real_equity, is_paper=is_paper)
    except Exception as exc:  # noqa: BLE001
        _log.error("forward_ab: real snapshot failed: %s", exc)

    try:
        scores = compute_shadow_scores(aggregated)
        _update_shadow_book(scores)
        shadow_eq = _shadow_equity()
        _record_snapshot("shadow", shadow_eq, is_paper=is_paper)
        _log.info(
            "forward_ab: real=%.2f shadow=%.2f (delta=%+.2f)",
            real_equity, shadow_eq, shadow_eq - real_equity,
        )
    except Exception as exc:  # noqa: BLE001
        _log.error("forward_ab: shadow computation failed: %s", exc)
