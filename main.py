"""Soros trading bot — main entry point.

Runs a single-threaded loop that:
  1. Collects OHLCV price data for the full universe (pinned ∪ watchlist)
  2. Screens the universe → selected_crypto ∪ selected_stocks (pinned always in)
  3. Computes deterministic signals for selected symbols
  4. Runs the sentiment pipeline when SENTIMENT_ENABLED=true
  5. Aggregates signals (blends sentiment into composite) for selected symbols
  6. Executes orders (paper or live, per CRYPTO_LIVE / STOCKS_LIVE toggles)
  7. Snapshots equity for drawdown tracking

Loop cadence is controlled by LOOP_INTERVAL_SECONDS (default: 3600 = 1 h).
Stop with SIGINT (Ctrl+C) or SIGTERM.
"""

from __future__ import annotations

import signal
import sys
import time

import config
from data import collector as crypto_collector
from data import stocks_collector
from database.db import get_connection, get_logger
from engine import order_executor, signal_aggregator, stocks_executor
from engine.risk_manager import RiskManager
from engine.screener import screen
from sentiment import runner as sentiment_runner
from signals import compute as signals_compute

_log = get_logger(__name__)

_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    _log.info("shutdown signal %d received — stopping after current cycle", signum)
    _shutdown = True


def _mark_to_market() -> None:
    """Update current_price and unrealized_pnl for all open positions."""
    conn = get_connection()
    open_rows = conn.execute(
        "SELECT id, symbol, side, quantity, entry_price FROM positions WHERE status = 'open'"
    ).fetchall()
    for row in open_rows:
        price_row = conn.execute(
            "SELECT close FROM prices WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            (row["symbol"],),
        ).fetchone()
        if price_row is None:
            continue
        current_price = float(price_row["close"])
        qty = float(row["quantity"])
        entry_price = float(row["entry_price"])
        if row["side"] == "long":
            unrealized = (current_price - entry_price) * qty
        else:
            unrealized = (entry_price - current_price) * qty
        conn.execute(
            "UPDATE positions SET current_price = ?, unrealized_pnl = ? WHERE id = ?",
            (current_price, unrealized, int(row["id"])),
        )
    conn.commit()


def _current_equity() -> float:
    """Compute mark-to-market equity: INITIAL_CAPITAL + realized P&L + unrealized P&L.

    Equity moves with the market every cycle after _mark_to_market() is called.
    Returns INITIAL_CAPITAL when no positions exist yet.
    """
    conn = get_connection()
    realized = float(conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0.0) AS t FROM positions WHERE status = 'closed'"
    ).fetchone()["t"])
    unrealized = float(conn.execute(
        "SELECT COALESCE(SUM(unrealized_pnl), 0.0) AS t FROM positions WHERE status = 'open'"
    ).fetchone()["t"])
    return config.INITIAL_CAPITAL + realized + unrealized


def _run_cycle(rm: RiskManager) -> None:
    """Execute one full bot cycle."""
    cycle_start = time.time()
    _log.info("=== cycle start ===")

    # 1. Collect price data for the full universe (pinned ∪ watchlist)
    _log.info("collecting crypto OHLCV")
    crypto_collector.collect_once()

    _log.info("collecting stocks OHLCV")
    stocks_collector.collect_once()

    # 2. Screen the universe — select symbols to operate this cycle.
    #    When SCREENER_ENABLED=False (default), returns pinned only.
    screener_result = screen()
    sel_crypto = screener_result.selected_crypto
    sel_stocks = screener_result.selected_stocks
    _log.info(
        "screener: crypto=%s stocks=%s (SCREENER_ENABLED=%s)",
        sel_crypto,
        sel_stocks,
        config.SCREENER_ENABLED,
    )

    # 3. Compute deterministic signals for selected symbols only
    _log.info("computing deterministic signals")
    signal_results = signals_compute.compute_once(
        crypto_symbols=sel_crypto,
        stock_symbols=sel_stocks,
    )

    det_scores: dict[str, float] = {r.symbol: r.composite_score for r in signal_results}

    # 4. Sentiment pipeline (best-effort; bot continues on failure)
    if config.SENTIMENT_ENABLED:
        _log.info("running sentiment pipeline")
        try:
            sentiment_runner.run(
                crypto_symbols=sel_crypto,
                stock_symbols=sel_stocks,
                deterministic_scores=det_scores,
            )
        except Exception as exc:
            _log.error("sentiment pipeline failed — continuing without sentiment: %s", exc)

    # 5. Aggregate signals (blends in sentiment from SQLite) for selected symbols
    _log.info("aggregating signals")
    aggregated = signal_aggregator.aggregate_once(
        crypto_symbols=sel_crypto,
        stock_symbols=sel_stocks,
    )

    # 6. Mark open positions to market, then compute equity
    _mark_to_market()
    is_paper = not (config.CRYPTO_LIVE or config.STOCKS_LIVE)
    equity = _current_equity()

    # 7. Execute orders
    _log.info("executing orders (equity=%.2f, paper=%s)", equity, is_paper)
    crypto_results = order_executor.execute_once(aggregated, equity)
    stocks_results = stocks_executor.execute_stocks_once(aggregated, equity)

    placed = len(crypto_results) + len(stocks_results)
    if placed:
        for r in crypto_results:
            _log.info("order: %s %s qty=%.6f price=%.4f paper=%s",
                      r.side, r.symbol, r.quantity, r.price, r.is_paper)
        for r in stocks_results:
            _log.info("order: %s %s qty=%.6f price=%.4f paper=%s",
                      r.side, r.symbol, r.quantity, r.price, r.is_paper)
    else:
        _log.info("no orders placed this cycle")

    # 8. Snapshot equity for drawdown tracking
    rm.record_equity(equity, is_paper=is_paper)

    elapsed = time.time() - cycle_start
    _log.info("=== cycle complete in %.1fs ===", elapsed)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _log.info("soros starting — validating config")
    config.validate_config()

    _log.info(
        "toggles: CRYPTO_LIVE=%s STOCKS_LIVE=%s SENTIMENT_ENABLED=%s SCREENER_ENABLED=%s",
        config.CRYPTO_LIVE,
        config.STOCKS_LIVE,
        config.SENTIMENT_ENABLED,
        config.SCREENER_ENABLED,
    )
    _log.info(
        "pinned: crypto=%s stocks=%s",
        config.CRYPTO_SYMBOLS,
        config.STOCK_SYMBOLS,
    )
    if config.SCREENER_ENABLED:
        _log.info(
            "watchlist: crypto=%s stocks=%s",
            config.CRYPTO_WATCHLIST,
            config.STOCK_WATCHLIST,
        )
    _log.info("loop interval: %ds", config.LOOP_INTERVAL_SECONDS)

    # Ensure the DB is initialised before the first cycle
    get_connection()

    rm = RiskManager()

    while not _shutdown:
        try:
            _run_cycle(rm)
        except Exception as exc:
            _log.error("cycle error: %s", exc, exc_info=True)

        if _shutdown:
            break

        _log.info("sleeping %ds until next cycle", config.LOOP_INTERVAL_SECONDS)
        # Sleep in small increments so SIGTERM/SIGINT is handled promptly
        deadline = time.monotonic() + config.LOOP_INTERVAL_SECONDS
        while not _shutdown and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    _log.info("soros stopped")


if __name__ == "__main__":
    sys.exit(main())
