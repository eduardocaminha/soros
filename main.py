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


def _current_equity(is_paper: bool) -> float:
    """Estimate current equity from open positions + any realised P&L baseline.

    When no equity snapshots exist, returns a default of 10 000 USD so the
    first cycle can size positions.  The risk manager records every value
    we pass here, so the drawdown gate starts tracking from the first cycle.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT equity FROM equity_curve ORDER BY ts DESC, id DESC LIMIT 1"
    ).fetchone()

    if row:
        return float(row["equity"])

    # First run: estimate from open positions
    open_rows = conn.execute(
        "SELECT entry_price, quantity FROM positions WHERE status = 'open'"
    ).fetchall()
    if open_rows:
        return sum(float(r["entry_price"]) * float(r["quantity"]) for r in open_rows)

    return 10_000.0


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

    # 5. Estimate equity for position sizing
    is_paper = not (config.CRYPTO_LIVE or config.STOCKS_LIVE)
    equity = _current_equity(is_paper)

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
