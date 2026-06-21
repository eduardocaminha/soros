"""A/B backtest command: sentiment OFF vs ON using Fear & Greed history.

Runs run_ab_backtest() over a user-specified price window and persists metrics
+ equity curves to the backtest_ab_results table so the dashboard can serve
them without recomputing on every page load.

This is an on-demand command — never called automatically.

Usage:
    python -m backtest.ab_command --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31
    python -m backtest.ab_command --screener --start 2024-01-01 --end 2024-12-31
    python -m backtest.ab_command --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31 \\
        --cache /tmp/fng_cache.json
"""

from __future__ import annotations

import json
import time
import uuid

import config
from backtest.ab_runner import run_ab_backtest
from backtest.engine import BacktestConfig, _load_prices_from_db
from backtest.sweep import _resolve_symbols
from sentiment.fear_greed_history import get_index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_ab_result(
    run_id: str,
    run_ts: int,
    cfg: BacktestConfig,
    ab_result,
) -> None:
    from database.db import get_connection

    conn = get_connection()
    symbols_json = json.dumps(list(cfg.symbols))
    off_equity_json = json.dumps(ab_result.off.equity_curve)
    on_equity_json = json.dumps(ab_result.on.equity_curve)

    conn.execute(
        """
        INSERT INTO backtest_ab_results (
            run_id, run_ts, start_ts, end_ts, symbols_json, fng_coverage_pct,
            off_total_return, off_cagr, off_sharpe, off_max_dd, off_win_rate, off_n_trades,
            on_total_return,  on_cagr,  on_sharpe,  on_max_dd,  on_win_rate,  on_n_trades,
            off_equity_json, on_equity_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, run_ts, cfg.start_ts, cfg.end_ts, symbols_json,
            ab_result.fng_coverage_pct,
            ab_result.off.total_return, ab_result.off.cagr, ab_result.off.sharpe,
            ab_result.off.max_drawdown, ab_result.off.win_rate, ab_result.off.num_trades,
            ab_result.on.total_return,  ab_result.on.cagr,  ab_result.on.sharpe,
            ab_result.on.max_drawdown,  ab_result.on.win_rate,  ab_result.on.num_trades,
            off_equity_json, on_equity_json,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ab_command(
    cfg_template: BacktestConfig,
    *,
    cache_path: str | None = None,
) -> None:
    """Resolve symbols, run the A/B backtest, persist to DB, print summary."""
    symbols = _resolve_symbols(cfg_template)
    cfg = BacktestConfig(
        symbols=symbols,
        start_ts=cfg_template.start_ts,
        end_ts=cfg_template.end_ts,
        initial_capital=cfg_template.initial_capital,
        use_screener=False,
        use_assembler=False,
    )

    prices_df = _load_prices_from_db(cfg)
    fng_index = get_index(cache_path=cache_path)

    run_id = uuid.uuid4().hex[:12]
    run_ts = int(time.time())

    ab = run_ab_backtest(cfg, fng_index, prices_df=prices_df)
    save_ab_result(run_id, run_ts, cfg, ab)
    _print_summary(run_id, cfg, ab)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_summary(run_id: str, cfg: BacktestConfig, ab) -> None:
    import datetime

    def _d(ts: int) -> str:
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    syms = ", ".join(f"{s}/{c}" for s, c in cfg.symbols)
    print(f"\n{'='*72}")
    print(f"  Backtest A/B — {syms}")
    print(f"  Período : {_d(cfg.start_ts)} → {_d(cfg.end_ts)}")
    print(f"  run_id  : {run_id}")
    print(f"  F&G coverage: {ab.fng_coverage_pct:.1%}")
    print(f"{'='*72}")
    print(
        f"  {'Métrica':<18}  {'Sem Sentimento':>16}  {'Com Sentimento (F&G)':>22}"
    )
    print(f"  {'-'*18}  {'-'*16}  {'-'*22}")
    _row = lambda label, off, on: print(f"  {label:<18}  {off:>16}  {on:>22}")
    _row(
        "Retorno Total",
        f"{ab.off.total_return:+.2%}",
        f"{ab.on.total_return:+.2%}",
    )
    _row("CAGR", f"{ab.off.cagr:+.2%}", f"{ab.on.cagr:+.2%}")
    _row("Sharpe", f"{ab.off.sharpe:.3f}", f"{ab.on.sharpe:.3f}")
    _row(
        "Max Drawdown",
        f"{ab.off.max_drawdown:.2%}",
        f"{ab.on.max_drawdown:.2%}",
    )
    _row(
        "Win Rate",
        f"{ab.off.win_rate:.2%}",
        f"{ab.on.win_rate:.2%}",
    )
    _row("Trades", str(ab.off.num_trades), str(ab.on.num_trades))
    print(f"{'='*72}\n")
    print("Resultados persistidos em backtest_ab_results.")


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
        description="Run soros backtest A/B: sentiment OFF vs ON (Fear & Greed history).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backtest.ab_command --symbols BTC/USDT:crypto "
            "--start 2024-01-01 --end 2024-12-31\n"
            "  python -m backtest.ab_command --screener "
            "--start 2024-01-01 --end 2024-12-31 --cache /tmp/fng.json"
        ),
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated SYMBOL:asset_class pairs.",
    )
    p.add_argument(
        "--screener",
        action="store_true",
        help="Derive symbol list from engine.screener.screen().",
    )
    p.add_argument(
        "--assembler",
        action="store_true",
        help="Build autonomous universe then screen.",
    )
    p.add_argument("--start", required=True, type=_parse_date, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, type=_parse_date, help="End date YYYY-MM-DD (UTC)")
    p.add_argument("--capital", type=float, default=None, help="Initial capital")
    p.add_argument("--cache", default=None, help="Path for F&G history JSON cache")
    p.add_argument("--db", default=None, help="Override DB_PATH")

    args = p.parse_args(argv)
    if not args.screener and not args.assembler and not args.symbols:
        p.error("--symbols is required unless --screener or --assembler is set")

    args.symbols = (
        [_parse_symbol(s.strip()) for s in args.symbols.split(",")]
        if args.symbols
        else []
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

    run_ab_command(cfg, cache_path=args.cache)


if __name__ == "__main__":
    main()
