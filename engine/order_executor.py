"""Crypto order executor — paper (dry_run) and live Binance.

Paper mode (CRYPTO_LIVE=false, default): orders are simulated in SQLite only.
Live mode  (CRYPTO_LIVE=true):           market orders sent to Binance via ccxt;
                                          actual fill price is used for cost basis.

RiskManager is always consulted before opening a position. Duplicate opens for
the same symbol are skipped (one open position per symbol enforced here).

Usage:
    executor = OrderExecutor()
    result = executor.execute(signal, equity=10_000.0)

    # Batch from a single cycle's aggregated signals
    results = execute_once(signals, equity=10_000.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ccxt

import config
from database.db import get_connection, get_logger
from engine.risk_manager import RiskManager

_log = get_logger(__name__)


@dataclass
class OrderResult:
    symbol: str
    asset_class: str
    side: str           # 'buy' | 'sell'
    quantity: float
    price: float
    is_paper: bool
    order_id: int       # SQLite orders.id
    position_id: int    # SQLite positions.id
    exchange_id: str | None  # Binance order ID; None for paper orders


# ---------------------------------------------------------------------------
# Private helpers — DB access
# ---------------------------------------------------------------------------

def _latest_close(symbol: str) -> float | None:
    """Most recent close price for *symbol* from the prices table; None if missing."""
    conn = get_connection()
    row = conn.execute(
        "SELECT close FROM prices WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return float(row["close"]) if row else None


def _get_open_position(symbol: str) -> Any | None:
    """Most recent open position row for *symbol*, or None."""
    conn = get_connection()
    return conn.execute(
        """
        SELECT id, quantity, entry_price, is_paper
        FROM positions
        WHERE symbol = ? AND status = 'open'
        ORDER BY opened_at DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()


def _signal_ts(signal_id: int) -> int | None:
    """Cycle timestamp of the signal row that triggered this order."""
    conn = get_connection()
    row = conn.execute("SELECT ts FROM signals WHERE id = ?", (signal_id,)).fetchone()
    return int(row["ts"]) if row else None


def _insert_position(
    symbol: str,
    asset_class: str,
    side: str,
    quantity: float,
    price: float,
    is_paper: bool,
) -> int:
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO positions
            (symbol, asset_class, side, quantity, entry_price, current_price, is_paper)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, asset_class, side, quantity, price, price, int(is_paper)),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def _insert_order(
    symbol: str,
    asset_class: str,
    side: str,
    quantity: float,
    price: float,
    is_paper: bool,
    exchange_id: str | None,
    position_id: int,
    sig_ts: int | None,
) -> int:
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO orders
            (symbol, asset_class, side, quantity, price, status,
             exchange_id, is_paper, position_id, signal_ts, filled_at)
        VALUES (?, ?, ?, ?, ?, 'filled', ?, ?, ?, ?, unixepoch())
        """,
        (symbol, asset_class, side, quantity, price,
         exchange_id, int(is_paper), position_id, sig_ts),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def _close_position_db(
    position_id: int, close_price: float, realized_pnl: float
) -> None:
    conn = get_connection()
    conn.execute(
        """
        UPDATE positions
        SET status = 'closed',
            current_price = ?,
            realized_pnl = ?,
            closed_at = unixepoch()
        WHERE id = ?
        """,
        (close_price, realized_pnl, position_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Private helper — Binance live execution
# ---------------------------------------------------------------------------

def _make_exchange() -> ccxt.binance:
    """Build a ccxt Binance spot instance using credentials from config."""
    return ccxt.binance(
        {
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OrderExecutor:
    """Executes crypto orders in paper or live mode.

    Paper mode is the default (CRYPTO_LIVE=false). In live mode orders are
    sent to Binance; LOT_SIZE rounding is applied automatically via ccxt.
    """

    def __init__(self, risk_manager: RiskManager | None = None) -> None:
        self._rm = risk_manager or RiskManager()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, signal: Any, equity: float) -> OrderResult | None:
        """Execute one order for *signal*.

        Args:
            signal:  AggregatedSignal from engine.signal_aggregator.
            equity:  Current account equity in USD (used for position sizing).

        Returns:
            OrderResult on a successful fill, None when no order was placed.
        """
        if signal.asset_class != "crypto":
            return None

        if signal.action == "buy":
            return self._open(signal, equity)
        if signal.action == "sell":
            return self._close(signal)
        return None  # hold

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def _open(self, signal: Any, equity: float) -> OrderResult | None:
        allowed, reason = self._rm.can_open(signal.symbol, signal.asset_class)
        if not allowed:
            _log.warning("risk block [%s buy]: %s", signal.symbol, reason)
            return None

        if _get_open_position(signal.symbol) is not None:
            _log.info("already open position for %s — skipping buy", signal.symbol)
            return None

        price = _latest_close(signal.symbol)
        if price is None or price <= 0.0:
            _log.warning("no price for %s — cannot size position", signal.symbol)
            return None

        size_usd = self._rm.position_size(equity)
        quantity = size_usd / price
        is_paper = not config.CRYPTO_LIVE
        exchange_id: str | None = None

        if config.CRYPTO_LIVE:
            exchange_id, price = self._place_live(signal.symbol, "buy", quantity, price)
            if exchange_id is None:
                return None

        pos_id = _insert_position(
            signal.symbol, signal.asset_class, "long", quantity, price, is_paper
        )
        order_id = _insert_order(
            signal.symbol, signal.asset_class, "buy", quantity, price,
            is_paper, exchange_id, pos_id, _signal_ts(signal.signal_id),
        )

        _log.info(
            "%s buy %s qty=%.6f price=%.4f",
            "paper" if is_paper else "LIVE",
            signal.symbol,
            quantity,
            price,
        )
        return OrderResult(
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            side="buy",
            quantity=quantity,
            price=price,
            is_paper=is_paper,
            order_id=order_id,
            position_id=pos_id,
            exchange_id=exchange_id,
        )

    def _close(self, signal: Any) -> OrderResult | None:
        pos = _get_open_position(signal.symbol)
        if pos is None:
            _log.info("no open position for %s — nothing to close", signal.symbol)
            return None

        price = _latest_close(signal.symbol)
        if price is None or price <= 0.0:
            _log.warning("no price for %s — cannot close position", signal.symbol)
            return None

        quantity = float(pos["quantity"])
        entry_price = float(pos["entry_price"])
        # Use the position's own is_paper flag, not the current toggle.  This
        # prevents sending a live sell for a paper-opened position when the
        # toggle is flipped mid-session, and vice-versa.
        is_paper = bool(pos["is_paper"])
        exchange_id: str | None = None

        if not is_paper:
            exchange_id, price = self._place_live(signal.symbol, "sell", quantity, price)
            if exchange_id is None:
                return None

        realized_pnl = (price - entry_price) * quantity
        pos_id = int(pos["id"])
        _close_position_db(pos_id, price, realized_pnl)
        order_id = _insert_order(
            signal.symbol, signal.asset_class, "sell", quantity, price,
            is_paper, exchange_id, pos_id, _signal_ts(signal.signal_id),
        )

        _log.info(
            "%s sell %s qty=%.6f price=%.4f pnl=%+.4f",
            "paper" if is_paper else "LIVE",
            signal.symbol,
            quantity,
            price,
            realized_pnl,
        )
        return OrderResult(
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            side="sell",
            quantity=quantity,
            price=price,
            is_paper=is_paper,
            order_id=order_id,
            position_id=pos_id,
            exchange_id=exchange_id,
        )

    # ------------------------------------------------------------------
    # Live exchange interaction
    # ------------------------------------------------------------------

    def _place_live(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fallback_price: float,
    ) -> tuple[str | None, float]:
        """Send a market order to Binance. Returns (exchange_id, fill_price).

        LOT_SIZE filter is applied via ccxt amount_to_precision().
        On any failure, logs the error and returns (None, fallback_price).
        """
        try:
            ex = _make_exchange()
            ex.load_markets()
            qty = float(ex.amount_to_precision(symbol, quantity))
            order = ex.create_market_order(symbol, side, qty)
            fill_price = float(
                order.get("average") or order.get("price") or fallback_price
            )
            exchange_id = str(order["id"])
            _log.info(
                "LIVE %s %s qty=%.8f fill=%.4f exchange_id=%s",
                side, symbol, qty, fill_price, exchange_id,
            )
            return exchange_id, fill_price
        except Exception as exc:
            _log.error(
                "live order failed [%s %s]: %s", side, symbol, exc, exc_info=True
            )
            return None, fallback_price


def execute_once(
    signals: list[Any],
    equity: float,
    executor: OrderExecutor | None = None,
) -> list[OrderResult]:
    """Execute orders for a batch of AggregatedSignals from one cycle.

    Args:
        signals:  Output of engine.signal_aggregator.aggregate_once().
        equity:   Current account equity in USD.
        executor: Pre-built OrderExecutor; a new one is created when omitted.

    Returns:
        List of OrderResult for every order that was placed (buy or sell).
    """
    ex = executor or OrderExecutor()
    results: list[OrderResult] = []
    for sig in signals:
        result = ex.execute(sig, equity)
        if result is not None:
            results.append(result)
    return results
