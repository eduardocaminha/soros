"""Stocks order executor — paper (dry_run) and live Alpaca.

Paper mode (STOCKS_LIVE=false, default): orders are simulated in SQLite only.
Live mode  (STOCKS_LIVE=true):           market orders sent to Alpaca at
                                          ALPACA_BASE_URL (defaults to the paper
                                          endpoint so live toggling can be tested
                                          safely before routing to the real endpoint).

RiskManager is always consulted before opening a position. Duplicate opens for
the same symbol are skipped (one open position per symbol enforced here).

Usage:
    executor = StocksExecutor()
    result = executor.execute(signal, equity=10_000.0)

    # Batch from a single cycle's aggregated signals
    results = execute_stocks_once(signals, equity=10_000.0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import config
from database.db import get_connection, get_logger
from engine.risk_manager import RiskManager

_log = get_logger(__name__)

_TIMEOUT = 15  # seconds per HTTP request


@dataclass
class StocksOrderResult:
    symbol: str
    asset_class: str
    side: str           # 'buy' | 'sell'
    quantity: float
    price: float
    is_paper: bool
    order_id: int       # SQLite orders.id
    position_id: int    # SQLite positions.id
    exchange_id: str | None  # Alpaca order ID; None for paper orders


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
# Private helper — Alpaca live execution
# ---------------------------------------------------------------------------

def _place_alpaca_order(
    symbol: str,
    side: str,
    quantity: float,
    fallback_price: float,
) -> tuple[str | None, float]:
    """Send a market order to Alpaca. Returns (exchange_id, fill_price).

    Routes to ALPACA_BASE_URL (defaults to the paper endpoint). On any
    failure, logs the error and returns (None, fallback_price) so the caller
    can abort without writing a partial DB record.
    """
    try:
        url = f"{config.ALPACA_BASE_URL}/v2/orders"
        payload = json.dumps({
            "symbol": symbol,
            "qty": str(round(quantity, 8)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }).encode()
        headers = {
            "APCA-API-KEY-ID": config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())

        exchange_id = str(data["id"])
        # Market orders may not have a fill price immediately; use fallback.
        fill_price = float(data.get("filled_avg_price") or fallback_price)
        _log.info(
            "LIVE %s %s qty=%.6f fill=%.4f exchange_id=%s",
            side, symbol, quantity, fill_price, exchange_id,
        )
        return exchange_id, fill_price
    except Exception as exc:
        _log.error(
            "alpaca order failed [%s %s]: %s", side, symbol, exc, exc_info=True
        )
        return None, fallback_price


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StocksExecutor:
    """Executes stock orders in paper or live Alpaca mode.

    Paper mode is the default (STOCKS_LIVE=false). In live mode orders are
    sent to Alpaca at ALPACA_BASE_URL (defaults to the paper endpoint).
    """

    def __init__(self, risk_manager: RiskManager | None = None) -> None:
        self._rm = risk_manager or RiskManager()

    def execute(self, signal: Any, equity: float) -> StocksOrderResult | None:
        """Execute one order for *signal*.

        Args:
            signal:  AggregatedSignal from engine.signal_aggregator.
            equity:  Current account equity in USD.

        Returns:
            StocksOrderResult on a successful fill, None when no order was placed.
        """
        if signal.asset_class != "stocks":
            return None

        if signal.action == "buy":
            return self._open(signal, equity)
        if signal.action == "sell":
            return self._close(signal)
        return None  # hold

    def _open(self, signal: Any, equity: float) -> StocksOrderResult | None:
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
        is_paper = not config.STOCKS_LIVE
        exchange_id: str | None = None

        if config.STOCKS_LIVE:
            exchange_id, price = _place_alpaca_order(signal.symbol, "buy", quantity, price)
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
        return StocksOrderResult(
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

    def _close(self, signal: Any) -> StocksOrderResult | None:
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
        is_paper = not config.STOCKS_LIVE
        exchange_id: str | None = None

        if config.STOCKS_LIVE:
            exchange_id, price = _place_alpaca_order(signal.symbol, "sell", quantity, price)
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
        return StocksOrderResult(
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


def execute_stocks_once(
    signals: list[Any],
    equity: float,
    executor: StocksExecutor | None = None,
) -> list[StocksOrderResult]:
    """Execute orders for a batch of AggregatedSignals from one cycle.

    Args:
        signals:  Output of engine.signal_aggregator.aggregate_once().
        equity:   Current account equity in USD.
        executor: Pre-built StocksExecutor; a new one is created when omitted.

    Returns:
        List of StocksOrderResult for every order that was placed (buy or sell).
    """
    ex = executor or StocksExecutor()
    results: list[StocksOrderResult] = []
    for sig in signals:
        result = ex.execute(sig, equity)
        if result is not None:
            results.append(result)
    return results
