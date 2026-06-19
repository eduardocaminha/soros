"""Sweep runner: backtest over a configurable parameter grid.

Runs ``run_backtest`` for each value in the grid, sharing a single price-load
and a single screener call, then stores per-value metrics in ``sweep_results``
so the dashboard can serve them without repeating the (expensive) backtest.

By default sweeps ``signal_threshold`` (the SIGNAL_THRESHOLD grid).  Pass a
``SweepSpec`` to sweep any other numeric ``BacktestConfig`` field — e.g.
``position_size_pct``, ``fee_pct`` — without rewriting the runner.

This is an on-demand command — never called on page load.

Usage:
    python -m backtest.sweep --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31
    python -m backtest.sweep --screener --start 2024-01-01 --end 2024-12-31
    python -m backtest.sweep --screener --start 2024-01-01 --end 2024-12-31 --thresholds 0.10,0.20,0.30
    python -m backtest.sweep --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31 \\
        --param position_size_pct --thresholds 0.05,0.10,0.15,0.20
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

import config
from backtest.engine import BacktestConfig, _load_prices_from_db, run_backtest


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SweepSpec:
    """Declares which ``BacktestConfig`` field to sweep and over what values.

    Extend the sweep to any numeric ``BacktestConfig`` field without rewriting
    ``run_sweep``.  The runner applies each value via ``dataclasses.replace``,
    keeping all other config fields fixed.

    Examples::

        SweepSpec("signal_threshold", [0.15, 0.20, 0.25, 0.30, 0.35])
        SweepSpec("position_size_pct", [0.05, 0.10, 0.15, 0.20])
        SweepSpec("fee_pct", [0.0005, 0.001, 0.002])
    """

    param: str
    values: list[float]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("SweepSpec.values must be non-empty")
        valid = {f.name for f in dataclasses.fields(BacktestConfig)}
        if self.param not in valid:
            raise ValueError(
                f"{self.param!r} is not a BacktestConfig field; valid: {sorted(valid)}"
            )


@dataclass
class SweepRow:
    """One row in sweep_results: backtest metrics for a single parameter value."""

    sweep_id: str
    run_ts: int
    signal_threshold: float  # swept value (param_name identifies which field)
    total_return: float
    cagr: float
    sharpe: float
    max_dd: float
    win_rate: float
    n_trades: int
    param_name: str = field(default="signal_threshold")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sweep(
    cfg_template: BacktestConfig,
    thresholds: Optional[Sequence[float]] = None,
    _prices_df: Optional[pd.DataFrame] = None,
    spec: Optional[SweepSpec] = None,
) -> list[SweepRow]:
    """Run backtest for each parameter value, persist results, return rows.

    Prices and the screener universe are resolved once before the loop so
    each iteration is a pure in-memory simulation.

    Args:
        cfg_template: Base config. The swept field is overridden per iteration;
            all other fields are preserved. ``use_screener`` / ``use_assembler``
            flags are resolved here then fixed for the loop.
        thresholds: Grid of ``signal_threshold`` values to sweep. Kept for
            backward compatibility; prefer ``spec=`` for new callers.
        _prices_df: Pre-loaded price DataFrame (for tests; skips DB load).
        spec: Declares which ``BacktestConfig`` field to sweep and over what
            values.  When provided, ``thresholds`` must be omitted.  Defaults
            to a ``signal_threshold`` sweep over ``config.SWEEP_THRESHOLDS``.

    Returns:
        List of SweepRow (one per value), already persisted to sweep_results.
    """
    if spec is not None and thresholds is not None:
        raise ValueError("Provide either 'thresholds' or 'spec', not both")

    if spec is None:
        _values: list[float] = (
            list(thresholds) if thresholds is not None else list(config.SWEEP_THRESHOLDS)
        )
        if not _values:
            raise ValueError("thresholds must be a non-empty sequence")
        spec = SweepSpec(param="signal_threshold", values=_values)

    # Resolve symbol list once — avoids repeated screener/API calls per value.
    symbols = _resolve_symbols(cfg_template)

    # Build a concrete base config with explicit symbols (no screener/assembler).
    base_cfg = dataclasses.replace(
        cfg_template,
        symbols=symbols,
        use_screener=False,
        use_assembler=False,
    )

    # Load prices once; shared across all iterations.
    if _prices_df is None:
        _prices_df = _load_prices_from_db(base_cfg)

    sweep_id = uuid.uuid4().hex[:12]
    run_ts = int(time.time())

    rows: list[SweepRow] = []
    for value in spec.values:
        cfg = dataclasses.replace(base_cfg, **{spec.param: value})
        result = run_backtest(cfg, prices_df=_prices_df)
        rows.append(
            SweepRow(
                sweep_id=sweep_id,
                run_ts=run_ts,
                signal_threshold=value,
                param_name=spec.param,
                total_return=result.total_return,
                cagr=result.cagr,
                sharpe=result.sharpe,
                max_dd=result.max_drawdown,
                win_rate=result.win_rate,
                n_trades=result.num_trades,
            )
        )

    _save_sweep_rows(rows)
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_symbols(cfg: BacktestConfig) -> list[tuple[str, str]]:
    """Return the effective symbol list, calling screener/assembler at most once."""
    if cfg.use_assembler:
        from data.assembler import assemble_universe
        from engine.screener import screen

        assembled = assemble_universe()
        result = screen(
            crypto_pinned=list(config.CRYPTO_SYMBOLS),
            crypto_watchlist=assembled.all_symbols,
            crypto_origins=assembled.origins,
        )
        return (
            [(sym, "crypto") for sym in result.selected_crypto]
            + [(sym, "stocks") for sym in result.selected_stocks]
        )

    if cfg.use_screener:
        from engine.screener import screen

        result = screen()
        return (
            [(sym, "crypto") for sym in result.selected_crypto]
            + [(sym, "stocks") for sym in result.selected_stocks]
        )

    return list(cfg.symbols)


def _save_sweep_rows(rows: list[SweepRow]) -> None:
    from database.db import get_connection

    conn = get_connection()
    conn.executemany(
        """
        INSERT INTO sweep_results
            (sweep_id, run_ts, signal_threshold, param_name, total_return, cagr, sharpe,
             max_dd, win_rate, n_trades)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.sweep_id, r.run_ts, r.signal_threshold, r.param_name,
                r.total_return, r.cagr, r.sharpe, r.max_dd, r.win_rate, r.n_trades,
            )
            for r in rows
        ],
    )
    conn.commit()


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
                f"Symbol must be 'SYMBOL:asset_class', got {s!r}"
            )
        sym, cls = s.split(":", 1)
        if cls not in ("crypto", "stocks"):
            raise argparse.ArgumentTypeError(
                f"asset_class must be 'crypto' or 'stocks', got {cls!r}"
            )
        return sym, cls

    p = argparse.ArgumentParser(
        description="Run soros sweep: backtest over a SIGNAL_THRESHOLD grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backtest.sweep --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31\n"
            "  python -m backtest.sweep --screener --start 2024-01-01 --end 2024-12-31\n"
            "  python -m backtest.sweep --screener --start 2024-01-01 --end 2024-12-31"
            " --thresholds 0.10,0.20,0.30"
        ),
    )
    p.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated SYMBOL:asset_class pairs, e.g. BTC/USDT:crypto. "
            "Required unless --screener or --assembler."
        ),
    )
    p.add_argument(
        "--screener",
        action="store_true",
        help="Derive symbol list from engine.screener.screen().",
    )
    p.add_argument(
        "--assembler",
        action="store_true",
        help=(
            "Build autonomous universe via data.assembler.assemble_universe(),"
            " then screen. Implies --screener."
        ),
    )
    p.add_argument("--start", required=True, type=_parse_date, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, type=_parse_date, help="End date YYYY-MM-DD (UTC)")
    p.add_argument("--capital", type=float, default=None, help="Initial capital")
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated values to sweep (default: config.SWEEP_THRESHOLDS)",
    )
    p.add_argument(
        "--param",
        default="signal_threshold",
        help=(
            "BacktestConfig field to sweep (default: signal_threshold). "
            "Examples: position_size_pct, fee_pct."
        ),
    )
    p.add_argument("--db", default=None, help="Override DB_PATH")

    args = p.parse_args(argv)
    if not args.screener and not args.assembler and not args.symbols:
        p.error("--symbols is required unless --screener or --assembler is set")

    args.symbols = (
        [_parse_symbol(s.strip()) for s in args.symbols.split(",")]
        if args.symbols
        else []
    )
    args.thresholds = (
        [float(x.strip()) for x in args.thresholds.split(",")]
        if args.thresholds
        else None
    )
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.db:
        import importlib
        import os

        os.environ["DB_PATH"] = args.db
        import database.db as _db_mod

        importlib.reload(_db_mod)

    cfg = BacktestConfig(
        symbols=args.symbols,
        start_ts=args.start,
        end_ts=args.end,
        initial_capital=args.capital,
        use_screener=args.screener,
        use_assembler=args.assembler,
    )

    sweep_values = args.thresholds  # None → SweepSpec defaults to config.SWEEP_THRESHOLDS
    if args.param != "signal_threshold":
        sweep_spec = SweepSpec(
            param=args.param,
            values=sweep_values if sweep_values else config.SWEEP_THRESHOLDS,
        )
        rows = run_sweep(cfg, spec=sweep_spec)
    else:
        rows = run_sweep(cfg, thresholds=sweep_values)

    param_name = rows[0].param_name
    current_threshold = config.SIGNAL_THRESHOLD
    param_label = param_name[:10]
    print(f"\n{'='*76}")
    print(
        f"  Sweep complete — {len(rows)} values  param={param_name}"
        f"  sweep_id={rows[0].sweep_id}"
    )
    print(
        f"  {param_label:>10}  {'Return':>8}  {'CAGR':>8}  "
        f"{'Sharpe':>7}  {'MaxDD':>7}  {'WinRate':>8}  {'Trades':>7}"
    )
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    for row in rows:
        marker = (
            " ← current"
            if param_name == "signal_threshold"
            and abs(row.signal_threshold - current_threshold) < 1e-9
            else ""
        )
        print(
            f"  {row.signal_threshold:>10.4f}  "
            f"{row.total_return:>+8.2%}  "
            f"{row.cagr:>+8.2%}  "
            f"{row.sharpe:>7.3f}  "
            f"{row.max_dd:>7.2%}  "
            f"{row.win_rate:>8.2%}  "
            f"{row.n_trades:>7d}"
            f"{marker}"
        )
    print(f"{'='*76}\n")
    print("Results persisted to sweep_results table.")


if __name__ == "__main__":
    main()
