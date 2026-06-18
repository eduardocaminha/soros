"""Soros trading bot — main entry point.

Runs a single-threaded loop that:
  1. Collects OHLCV price data (crypto via Binance/ccxt, stocks via Alpaca/yfinance)
  2. Computes deterministic signals (momentum + volatility + funding_rate)
  3. Runs the sentiment pipeline when SENTIMENT_ENABLED=true
  4. Aggregates signals (blends sentiment into composite)
  5. Executes orders (paper or live, per CRYPTO_LIVE / STOCKS_LIVE toggles)
  6. Snapshots equity for drawdown tracking

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
from sentiment import runner as sentiment_runner
from signals import compute as signals_compute

_log = get_logger(__name__)

_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    _log.info("shutdown signal %d received — stopping after current cycle", signum)
    _shutdown = True


def _current_equity() -> float:
    """Mark-to-market equity: INITIAL_CAPITAL + realized_pnl + unrealized_pnl.

    Updates current_price and unrealized_pnl for every open position from the
    latest prices row before summing, so the equity curve moves with the market.
    Falls back to INITIAL_CAPITAL when there are no positions yet.
    """
    conn = get_connection()

    # Mark open positions to market
    open_positions = conn.execute(
        "SELECT id, symbol, quantity, entry_price FROM positions WHERE status = 'open'"
    ).fetchall()

    for pos in open_positions:
        row = conn.execute(
            "SELECT close FROM prices WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            (pos["symbol"],),
        ).fetchone()
        if row:
            current_price = float(row["close"])
            unrealized_pnl = (
                (current_price - float(pos["entry_price"])) * float(pos["quantity"])
            )
            conn.execute(
                "UPDATE positions SET current_price = ?, unrealized_pnl = ? WHERE id = ?",
                (current_price, unrealized_pnl, pos["id"]),
            )
    conn.commit()

    row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0.0) AS total"
        " FROM positions WHERE status = 'closed'"
    ).fetchone()
    total_realized = float(row["total"])

    row = conn.execute(
        "SELECT COALESCE(SUM(unrealized_pnl), 0.0) AS total"
        " FROM positions WHERE status = 'open'"
    ).fetchone()
    total_unrealized = float(row["total"])

    return config.INITIAL_CAPITAL + total_realized + total_unrealized


def _run_cycle(rm: RiskManager) -> None:
    """Execute one full bot cycle."""
    cycle_start = time.time()
    _log.info("=== cycle start ===")

    # 1. Collect price data
    _log.info("collecting crypto OHLCV")
    crypto_collector.collect_once()

    _log.info("collecting stocks OHLCV")
    stocks_collector.collect_once()

    # 2. Compute deterministic signals
    _log.info("computing deterministic signals")
    signal_results = signals_compute.compute_once()

    det_scores: dict[str, float] = {r.symbol: r.composite_score for r in signal_results}

    # 3. Sentiment pipeline (best-effort; bot continues on failure)
    if config.SENTIMENT_ENABLED:
        _log.info("running sentiment pipeline")
        try:
            sentiment_runner.run(
                deterministic_scores=det_scores,
            )
        except Exception as exc:
            _log.error("sentiment pipeline failed — continuing without sentiment: %s", exc)

    # 4. Aggregate signals (blends in sentiment from SQLite)
    _log.info("aggregating signals")
    aggregated = signal_aggregator.aggregate_once()

    # 5. Mark-to-market and compute current equity for position sizing
    is_paper = not (config.CRYPTO_LIVE or config.STOCKS_LIVE)
    equity = _current_equity()

    # 6. Execute orders
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

    # 7. Snapshot equity for drawdown tracking
    rm.record_equity(equity, is_paper=is_paper)

    elapsed = time.time() - cycle_start
    _log.info("=== cycle complete in %.1fs ===", elapsed)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _log.info("soros starting — validating config")
    config.validate_config()

    _log.info(
        "toggles: CRYPTO_LIVE=%s STOCKS_LIVE=%s SENTIMENT_ENABLED=%s",
        config.CRYPTO_LIVE,
        config.STOCKS_LIVE,
        config.SENTIMENT_ENABLED,
    )
    _log.info(
        "symbols: crypto=%s stocks=%s",
        config.CRYPTO_SYMBOLS,
        config.STOCK_SYMBOLS,
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
