"""Hard risk limits for the soros trading bot (non-bypassable).

Enforces two hard stops that cannot be relaxed at runtime:
  - Drawdown gate: if peak-to-trough drawdown >= MAX_DRAWDOWN_PCT (15 %),
    no new positions may be opened.
  - Position cap: total open positions across both asset classes must stay
    below MAX_OPEN_POSITIONS (3).

Position sizing is also centralised here so all executors use the same logic.

Usage:
    rm = RiskManager()
    allowed, reason = rm.can_open("BTC/USDT", "crypto")
    if allowed:
        size = rm.position_size(equity=10_000)
    rm.record_equity(equity=10_000, is_paper=True)
"""

from __future__ import annotations

import config
from database.db import get_connection, get_logger

_log = get_logger(__name__)

_MAX_DRAWDOWN = config.MAX_DRAWDOWN_PCT    # 0.15 — hard limit
_MAX_POSITIONS = config.MAX_OPEN_POSITIONS  # 3    — hard limit


class RiskManager:
    """Stateless risk gate; all persistent state lives in SQLite."""

    def can_open(self, symbol: str, asset_class: str) -> tuple[bool, str]:
        """Return (allowed, reason) for opening a new position in *symbol*.

        Blocked when drawdown >= MAX_DRAWDOWN_PCT or open positions >= MAX_OPEN_POSITIONS.
        """
        dd = self._current_drawdown()
        if dd >= _MAX_DRAWDOWN:
            reason = f"drawdown {dd:.1%} >= limit {_MAX_DRAWDOWN:.1%}"
            _log.warning("risk block [%s %s]: %s", symbol, asset_class, reason)
            return False, reason

        open_count = self._open_position_count()
        if open_count >= _MAX_POSITIONS:
            reason = f"open positions {open_count} >= limit {_MAX_POSITIONS}"
            _log.warning("risk block [%s %s]: %s", symbol, asset_class, reason)
            return False, reason

        return True, ""

    def position_size(self, equity: float) -> float:
        """Dollar amount to allocate for one new position."""
        return equity * config.POSITION_SIZE_PCT

    def record_equity(self, equity: float, is_paper: bool = True) -> None:
        """Snapshot current equity; update running peak and drawdown in equity_curve."""
        conn = get_connection()
        row = conn.execute(
            "SELECT peak_equity FROM equity_curve ORDER BY ts DESC, id DESC LIMIT 1"
        ).fetchone()

        peak = row["peak_equity"] if row else equity
        if equity > peak:
            peak = equity

        drawdown_pct = (peak - equity) / peak if peak > 0.0 else 0.0

        conn.execute(
            """
            INSERT INTO equity_curve (equity, peak_equity, drawdown_pct, is_paper)
            VALUES (?, ?, ?, ?)
            """,
            (equity, peak, drawdown_pct, int(is_paper)),
        )
        conn.commit()
        _log.info(
            "equity snapshot: equity=%.2f peak=%.2f drawdown=%.1f%%",
            equity,
            peak,
            drawdown_pct * 100,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_drawdown(self) -> float:
        """Most recent drawdown_pct from equity_curve, or 0.0 when no history."""
        conn = get_connection()
        row = conn.execute(
            "SELECT drawdown_pct FROM equity_curve ORDER BY ts DESC, id DESC LIMIT 1"
        ).fetchone()
        return float(row["drawdown_pct"]) if row else 0.0

    def _open_position_count(self) -> int:
        """Count all currently open positions (both asset classes)."""
        conn = get_connection()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM positions WHERE status = 'open'"
        ).fetchone()
        return int(row["cnt"]) if row else 0
