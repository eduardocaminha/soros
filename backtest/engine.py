"""Backtest harness for the soros trading strategy.

Replays historical OHLCV from the prices table through the live deterministic
signal pipeline (momentum + volatility + funding), simulates the same
paper-execution cycle (fees, slippage, mark-to-market, risk limits), and
reports:

    total_return, CAGR, Sharpe, max_drawdown, win_rate, num_trades

Sentiment is excluded from backtest signals: no LLM replay is possible.
The deterministic composite is computed with the same _deterministic_composite
logic from signals.compute (weights re-normalised to exclude sentiment).

Usage:
    from backtest.engine import BacktestConfig, run_backtest, print_result
    cfg = BacktestConfig(
        symbols=[("BTC/USDT", "crypto"), ("AAPL", "stocks")],
        start_ts=1700000000,
        end_ts=1710000000,
    )
    result = run_backtest(cfg)
    print_result(result)

CLI:
    python -m backtest --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import config
from signals import funding, momentum, volatility
from signals.compute import _action, _deterministic_composite


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    symbols: list[tuple[str, str]]  # [(symbol, asset_class), ...]
    start_ts: int                   # unix epoch seconds, inclusive
    end_ts: int                     # unix epoch seconds, inclusive
    initial_capital: float = field(default=None)   # type: ignore[assignment]
    fee_pct: float = field(default=None)            # type: ignore[assignment]
    slippage_pct: float = field(default=None)       # type: ignore[assignment]
    position_size_pct: float = field(default=None)  # type: ignore[assignment]
    max_open_positions: int = field(default=None)   # type: ignore[assignment]
    max_drawdown_pct: float = field(default=None)   # type: ignore[assignment]
    signal_threshold: float = field(default=None)   # type: ignore[assignment]
    window: int = 200  # candle lookback for signal computation

    def __post_init__(self) -> None:
        if self.initial_capital is None:
            self.initial_capital = config.INITIAL_CAPITAL
        if self.fee_pct is None:
            self.fee_pct = config.FEE_PCT
        if self.slippage_pct is None:
            self.slippage_pct = config.SLIPPAGE_PCT
        if self.position_size_pct is None:
            self.position_size_pct = config.POSITION_SIZE_PCT
        if self.max_open_positions is None:
            self.max_open_positions = config.MAX_OPEN_POSITIONS
        if self.max_drawdown_pct is None:
            self.max_drawdown_pct = config.MAX_DRAWDOWN_PCT
        if self.signal_threshold is None:
            self.signal_threshold = config.SIGNAL_THRESHOLD


@dataclass
class Trade:
    symbol: str
    asset_class: str
    entry_ts: int
    exit_ts: Optional[int]     # None while position is open
    entry_price: float         # after fees/slippage
    exit_price: Optional[float]  # after fees/slippage; None while open
    quantity: float
    realized_pnl: float        # 0.0 while open


@dataclass
class BacktestResult:
    cfg: BacktestConfig
    initial_capital: float
    final_equity: float
    total_return: float        # e.g. 0.15 = 15 %
    cagr: float
    sharpe: float
    max_drawdown: float        # e.g. 0.12 = 12 %
    win_rate: float            # closed trades only
    num_trades: int            # total closed trades
    num_wins: int
    equity_curve: list[tuple[int, float]]  # [(ts, equity), ...]
    trades: list[Trade]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_prices_from_db(cfg: BacktestConfig) -> pd.DataFrame:
    """Load OHLCV rows for all symbols covering the backtest range plus warmup.

    Fetches from start_ts - window * 3600 s to end_ts so that signal
    computation has enough lookback at the first candle of the range.
    """
    from database.db import get_connection

    warmup = cfg.window * 3600  # seconds of extra history for signal warmup
    fetch_from = cfg.start_ts - warmup

    conn = get_connection()
    placeholders = ",".join("?" for _ in cfg.symbols)
    sym_list = [s for s, _ in cfg.symbols]

    rows = conn.execute(
        f"""
        SELECT symbol, asset_class, ts, open, high, low, close, volume, funding_rate
        FROM prices
        WHERE symbol IN ({placeholders})
          AND timeframe = ?
          AND ts BETWEEN ? AND ?
        ORDER BY ts ASC
        """,
        (*sym_list, config.OHLCV_TIMEFRAME, fetch_from, cfg.end_ts),
    ).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=["symbol", "asset_class", "ts", "open", "high", "low",
                     "close", "volume", "funding_rate"]
        )

    return pd.DataFrame(
        rows,
        columns=["symbol", "asset_class", "ts", "open", "high", "low",
                 "close", "volume", "funding_rate"],
    )


def _get_signal(
    window_df: pd.DataFrame,
    asset_class: str,
    signal_threshold: float,
) -> tuple[float, str]:
    """Compute deterministic composite + action from an OHLCV window."""
    if window_df.empty:
        return 0.0, "hold"

    closes = window_df["close"].astype(float)
    latest_funding = (
        window_df["funding_rate"].iloc[-1] if asset_class == "crypto" else None
    )

    mom = momentum.compute(closes)
    vol = volatility.compute(window_df)
    fund = funding.compute(latest_funding) if asset_class == "crypto" else None

    composite = _deterministic_composite(mom, vol, fund, asset_class)

    # Respect the caller's threshold rather than the global config default.
    if composite >= signal_threshold:
        action = "buy"
    elif composite <= -signal_threshold:
        action = "sell"
    else:
        action = "hold"

    return composite, action


def _compute_metrics(
    equity_curve: list[tuple[int, float]],
    initial_capital: float,
    trades: list[Trade],
) -> tuple[float, float, float, float, float, float, int, int]:
    """Return (final_equity, total_return, cagr, sharpe, max_drawdown, win_rate, num_trades, num_wins)."""
    if not equity_curve:
        return initial_capital, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0

    final_equity = equity_curve[-1][1]
    total_return = (final_equity / initial_capital) - 1.0

    # CAGR — requires at least 1 day to avoid exponent overflow
    start_ts = equity_curve[0][0]
    end_ts = equity_curve[-1][0]
    years = (end_ts - start_ts) / (365.25 * 24 * 3600)
    _MIN_YEARS = 1.0 / 365.0
    if years >= _MIN_YEARS and final_equity > 0.0 and initial_capital > 0.0:
        cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Sharpe (annualised, risk-free rate = 0)
    equities = [e for _, e in equity_curve]
    returns = [
        equities[i] / equities[i - 1] - 1.0
        for i in range(1, len(equities))
        if equities[i - 1] != 0.0
    ]
    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r)
        if std_r > 0.0 and years > 0.0:
            periods_per_year = len(equity_curve) / years
            sharpe = mean_r / std_r * math.sqrt(periods_per_year)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0.0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Win rate (closed trades only)
    closed = [t for t in trades if t.exit_ts is not None]
    num_trades = len(closed)
    num_wins = sum(1 for t in closed if t.realized_pnl > 0.0)
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0

    return final_equity, total_return, cagr, sharpe, max_dd, win_rate, num_trades, num_wins


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    cfg: BacktestConfig,
    prices_df: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Run the backtest simulation.

    Args:
        cfg: Backtest configuration (symbols, date range, capital, risk params).
        prices_df: Optional pre-loaded price DataFrame with columns
            [symbol, asset_class, ts, open, high, low, close, volume, funding_rate].
            If None, data is loaded from the configured SQLite DB.

    Returns:
        BacktestResult with all metrics and the full equity curve + trade list.
    """
    if prices_df is None:
        prices_df = _load_prices_from_db(cfg)

    # All unique timestamps within the backtest range, sorted ascending.
    in_range = prices_df[prices_df["ts"].between(cfg.start_ts, cfg.end_ts)]
    timestamps = sorted(in_range["ts"].unique())

    if not timestamps:
        return BacktestResult(
            cfg=cfg,
            initial_capital=cfg.initial_capital,
            final_equity=cfg.initial_capital,
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            num_trades=0,
            num_wins=0,
            equity_curve=[],
            trades=[],
        )

    # Mutable state
    positions: dict[str, dict] = {}   # symbol -> {asset_class, quantity, entry_price, entry_ts}
    trades: list[Trade] = []
    realized_pnl = 0.0
    peak_equity = cfg.initial_capital

    equity_curve: list[tuple[int, float]] = []

    # Pre-group prices by symbol for O(1) window slicing.
    by_symbol: dict[str, pd.DataFrame] = {
        sym: prices_df[prices_df["symbol"] == sym].sort_values("ts").reset_index(drop=True)
        for sym, _ in cfg.symbols
    }

    for ts in timestamps:
        # Current close for each symbol at this candle.
        current_closes: dict[str, float] = {}
        for sym, _ in cfg.symbols:
            sym_df = by_symbol.get(sym)
            if sym_df is None or sym_df.empty:
                continue
            row = sym_df[sym_df["ts"] == ts]
            if not row.empty:
                current_closes[sym] = float(row["close"].iloc[0])

        # Mark-to-market: unrealized P&L on open positions.
        unrealized_pnl = 0.0
        for sym, pos in positions.items():
            price = current_closes.get(sym, pos["entry_price"])
            unrealized_pnl += (price - pos["entry_price"]) * pos["quantity"]

        equity = cfg.initial_capital + realized_pnl + unrealized_pnl
        if equity > peak_equity:
            peak_equity = equity
        equity_curve.append((ts, equity))

        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 0.0

        # Process each symbol.
        for sym, asset_class in cfg.symbols:
            close = current_closes.get(sym)
            if close is None:
                continue

            # Signal: use all candles up to and including this timestamp.
            sym_df = by_symbol.get(sym)
            if sym_df is None:
                continue
            window_df = sym_df[sym_df["ts"] <= ts].tail(cfg.window).copy()
            _, action = _get_signal(window_df, asset_class, cfg.signal_threshold)

            if action == "buy" and sym not in positions:
                if len(positions) >= cfg.max_open_positions:
                    continue
                if drawdown >= cfg.max_drawdown_pct:
                    continue

                entry_price = close * (1.0 + cfg.fee_pct + cfg.slippage_pct)
                quantity = (equity * cfg.position_size_pct) / close
                positions[sym] = {
                    "asset_class": asset_class,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "entry_ts": ts,
                }
                trades.append(
                    Trade(
                        symbol=sym,
                        asset_class=asset_class,
                        entry_ts=ts,
                        exit_ts=None,
                        entry_price=entry_price,
                        exit_price=None,
                        quantity=quantity,
                        realized_pnl=0.0,
                    )
                )

            elif action == "sell" and sym in positions:
                pos = positions.pop(sym)
                exit_price = close * (1.0 - cfg.fee_pct - cfg.slippage_pct)
                pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                realized_pnl += pnl

                for trade in reversed(trades):
                    if trade.symbol == sym and trade.exit_ts is None:
                        trade.exit_ts = ts
                        trade.exit_price = exit_price
                        trade.realized_pnl = pnl
                        break

    # Mark any remaining open positions as closed at the last available price.
    last_ts = timestamps[-1]
    for sym, pos in list(positions.items()):
        sym_df = by_symbol.get(sym)
        last_row = (
            sym_df[sym_df["ts"] <= last_ts].tail(1) if sym_df is not None else None
        )
        if last_row is None or last_row.empty:
            last_close = pos["entry_price"]
        else:
            last_close = float(last_row["close"].iloc[0])

        exit_price = last_close * (1.0 - cfg.fee_pct - cfg.slippage_pct)
        pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        realized_pnl += pnl

        for trade in reversed(trades):
            if trade.symbol == sym and trade.exit_ts is None:
                trade.exit_ts = last_ts
                trade.exit_price = exit_price
                trade.realized_pnl = pnl
                break

    positions.clear()

    final_equity, total_return, cagr, sharpe, max_dd, win_rate, num_trades, num_wins = (
        _compute_metrics(equity_curve, cfg.initial_capital, trades)
    )

    return BacktestResult(
        cfg=cfg,
        initial_capital=cfg.initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=num_trades,
        num_wins=num_wins,
        equity_curve=equity_curve,
        trades=trades,
    )


def print_result(result: BacktestResult, *, show_trades: bool = False) -> None:
    """Print a formatted summary table to stdout."""
    import datetime

    def _fmt_ts(ts: int) -> str:
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    start = _fmt_ts(result.cfg.start_ts)
    end = _fmt_ts(result.cfg.end_ts)
    symbols = ", ".join(f"{s}/{c}" for s, c in result.cfg.symbols)

    print(f"\n{'='*56}")
    print(f"  Backtest: {symbols}")
    print(f"  Period  : {start} → {end}")
    print(f"{'='*56}")
    print(f"  Initial capital : ${result.initial_capital:>12,.2f}")
    print(f"  Final equity    : ${result.final_equity:>12,.2f}")
    print(f"  Total return    : {result.total_return:>+11.2%}")
    print(f"  CAGR            : {result.cagr:>+11.2%}")
    print(f"  Sharpe ratio    : {result.sharpe:>12.3f}")
    print(f"  Max drawdown    : {result.max_drawdown:>11.2%}")
    print(f"  Win rate        : {result.win_rate:>11.2%}  ({result.num_wins}/{result.num_trades} trades)")
    print(f"{'='*56}\n")

    if show_trades and result.trades:
        import datetime

        print(f"  {'Symbol':<14} {'Entry':<12} {'Exit':<12} {'PnL':>10}")
        print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*10}")
        for t in result.trades:
            entry = _fmt_ts(t.entry_ts)
            exit_ = _fmt_ts(t.exit_ts) if t.exit_ts else "open"
            print(f"  {t.symbol:<14} {entry:<12} {exit_:<12} {t.realized_pnl:>+10.2f}")
        print()


def save_result(
    result: BacktestResult,
    path: str,
    fmt: str = "csv",
) -> None:
    """Save equity curve and trade list to *path* in 'csv' or 'json' format."""
    import json

    equity_df = pd.DataFrame(result.equity_curve, columns=["ts", "equity"])
    trades_data = [
        {
            "symbol": t.symbol,
            "asset_class": t.asset_class,
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "realized_pnl": t.realized_pnl,
        }
        for t in result.trades
    ]

    if fmt == "csv":
        base = path.removesuffix(".csv")
        equity_df.to_csv(f"{base}_equity.csv", index=False)
        pd.DataFrame(trades_data).to_csv(f"{base}_trades.csv", index=False)
        print(f"Saved: {base}_equity.csv, {base}_trades.csv")
    elif fmt == "json":
        summary = {
            "total_return": result.total_return,
            "cagr": result.cagr,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "num_trades": result.num_trades,
            "num_wins": result.num_wins,
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "equity_curve": result.equity_curve,
            "trades": trades_data,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {path}")
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'csv' or 'json'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None):
    import argparse
    import datetime

    def _parse_date(s: str) -> int:
        dt = datetime.datetime.strptime(s, "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc
        )
        return int(dt.timestamp())

    def _parse_symbol(s: str) -> tuple[str, str]:
        if ":" not in s:
            raise argparse.ArgumentTypeError(
                f"Symbol must be in 'SYMBOL:asset_class' form, got {s!r}"
            )
        sym, cls = s.split(":", 1)
        if cls not in ("crypto", "stocks"):
            raise argparse.ArgumentTypeError(
                f"asset_class must be 'crypto' or 'stocks', got {cls!r}"
            )
        return sym, cls

    p = argparse.ArgumentParser(
        description="Run soros backtest over historical prices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backtest --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31\n"
            "  python -m backtest --symbols BTC/USDT:crypto,AAPL:stocks --start 2024-06-01 "
            "--capital 50000 --output json --out results.json"
        ),
    )
    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated SYMBOL:asset_class pairs, e.g. BTC/USDT:crypto,AAPL:stocks",
    )
    p.add_argument("--start", required=True, type=_parse_date, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, type=_parse_date, help="End date YYYY-MM-DD (UTC)")
    p.add_argument("--capital", type=float, default=None, help="Initial capital (default: INITIAL_CAPITAL env)")
    p.add_argument("--output", choices=["csv", "json"], default=None, help="Save results in this format")
    p.add_argument("--out", default="backtest_result", help="Output file stem (default: backtest_result)")
    p.add_argument("--trades", action="store_true", help="Print individual trades in summary")
    p.add_argument("--db", default=None, help="Override DB_PATH")

    args = p.parse_args(argv)
    args.symbols = [_parse_symbol(s.strip()) for s in args.symbols.split(",")]
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.db:
        import os
        os.environ["DB_PATH"] = args.db
        # Re-initialise DB connection with new path.
        import importlib
        import database.db as _db_mod
        importlib.reload(_db_mod)

    cfg = BacktestConfig(
        symbols=args.symbols,
        start_ts=args.start,
        end_ts=args.end,
        initial_capital=args.capital,  # None → __post_init__ fills from config
    )

    result = run_backtest(cfg)
    print_result(result, show_trades=args.trades)

    if args.output:
        ext = "json" if args.output == "json" else "csv"
        out_path = args.out if args.out.endswith(f".{ext}") else f"{args.out}.{ext}"
        save_result(result, out_path, fmt=args.output)


if __name__ == "__main__":
    main()
