"""Paper trading validation for the soros bot.

Reads the SQLite database and checks whether 48 h+ of continuous paper
trading has been completed with valid data.  Intended to be run once
before enabling live execution (CRYPTO_LIVE / STOCKS_LIVE).

Exit codes:
    0 — validation passed
    1 — validation failed (see printed report for details)

Usage:
    python validate_paper.py
    python validate_paper.py --db /path/to/soros.db
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import config
from database.db import get_connection

_REQUIRED_HOURS: float = 48.0
_REQUIRED_SECONDS: float = _REQUIRED_HOURS * 3600


@dataclass
class PaperValidationReport:
    """Summary of paper trading validation results."""

    passed: bool
    duration_hours: float
    cycles_run: int
    signals_generated: int
    orders_placed: int
    realized_pnl: float
    max_drawdown_pct: float
    live_order_leak: bool  # True means live orders were found — should never happen
    failures: list[str] = field(default_factory=list)


def validate(db_path: str | None = None) -> PaperValidationReport:
    """Run all paper trading validation checks and return a report.

    Uses *db_path* when provided; otherwise falls back to config.DB_PATH.
    Does not raise — all failures are collected into the returned report.
    """
    if db_path is not None:
        import sqlite3
        from pathlib import Path
        from database import db as db_module

        schema = (Path(__file__).parent / "database" / "schema.sql").read_text()

        class _FixedPathDB:
            def connect(self):
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(schema)
                return conn

        db_module._db = _FixedPathDB()

    conn = get_connection()
    failures: list[str] = []

    # ── 1. Duration check ──────────────────────────────────────────────────
    row = conn.execute(
        "SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS cnt "
        "FROM equity_curve WHERE is_paper = 1"
    ).fetchone()

    first_ts = row["first_ts"]
    last_ts = row["last_ts"]
    cycles_run = int(row["cnt"]) if row["cnt"] else 0

    if first_ts is None or last_ts is None:
        duration_secs = 0.0
        failures.append("no paper equity snapshots found — bot has not run yet")
    else:
        duration_secs = float(last_ts - first_ts)

    duration_hours = duration_secs / 3600.0

    if duration_hours < _REQUIRED_HOURS and first_ts is not None:
        failures.append(
            f"paper run duration {duration_hours:.1f} h < required {_REQUIRED_HOURS:.0f} h"
        )

    # ── 2. Signals check ───────────────────────────────────────────────────
    sig_row = conn.execute("SELECT COUNT(*) AS cnt FROM signals").fetchone()
    signals_generated = int(sig_row["cnt"]) if sig_row else 0

    if signals_generated == 0:
        failures.append("no deterministic signals recorded — signals pipeline may not have run")

    # ── 3. Orders check ────────────────────────────────────────────────────
    ord_row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN is_paper = 0 THEN 1 ELSE 0 END) AS live "
        "FROM orders"
    ).fetchone()

    orders_placed = int(ord_row["total"]) if ord_row else 0
    live_order_count = int(ord_row["live"]) if (ord_row and ord_row["live"] is not None) else 0
    live_order_leak = live_order_count > 0

    if live_order_leak:
        failures.append(
            f"{live_order_count} live order(s) found — CRYPTO_LIVE / STOCKS_LIVE were on during paper period"
        )

    # ── 4. P&L and drawdown ────────────────────────────────────────────────
    pnl_row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0.0) AS pnl "
        "FROM positions WHERE is_paper = 1"
    ).fetchone()
    realized_pnl = float(pnl_row["pnl"]) if pnl_row else 0.0

    dd_row = conn.execute(
        "SELECT COALESCE(MAX(drawdown_pct), 0.0) AS max_dd "
        "FROM equity_curve WHERE is_paper = 1"
    ).fetchone()
    max_drawdown_pct = float(dd_row["max_dd"]) if dd_row else 0.0

    if max_drawdown_pct >= config.MAX_DRAWDOWN_PCT:
        failures.append(
            f"max drawdown {max_drawdown_pct:.1%} reached the {config.MAX_DRAWDOWN_PCT:.0%} "
            "hard limit — bot stopped placing orders at that point (expected behaviour, but "
            "review the equity curve before enabling live trading)"
        )

    passed = len(failures) == 0

    return PaperValidationReport(
        passed=passed,
        duration_hours=duration_hours,
        cycles_run=cycles_run,
        signals_generated=signals_generated,
        orders_placed=orders_placed,
        realized_pnl=realized_pnl,
        max_drawdown_pct=max_drawdown_pct,
        live_order_leak=live_order_leak,
        failures=failures,
    )


def print_report(report: PaperValidationReport) -> None:
    """Print a human-readable validation report."""
    status = "PASSED" if report.passed else "FAILED"
    print(f"\n{'=' * 50}")
    print(f"  Paper Trading Validation: {status}")
    print(f"{'=' * 50}")
    print(f"  Duration          : {report.duration_hours:.1f} h  (required: {_REQUIRED_HOURS:.0f} h)")
    print(f"  Equity snapshots  : {report.cycles_run}")
    print(f"  Signals generated : {report.signals_generated}")
    print(f"  Orders placed     : {report.orders_placed}")
    print(f"  Realized P&L      : ${report.realized_pnl:+.2f}")
    print(f"  Max drawdown      : {report.max_drawdown_pct:.1%}")
    print(f"  Live order leak   : {'YES (!) ' if report.live_order_leak else 'no'}")

    if report.failures:
        print(f"\n  Failures ({len(report.failures)}):")
        for f in report.failures:
            print(f"    - {f}")

    print()

    if report.passed:
        print("  Ready to enable live trading.")
        print("  Set CRYPTO_LIVE=true and/or STOCKS_LIVE=true when ready.")
    else:
        print("  Do NOT enable live trading until all failures are resolved.")

    print(f"{'=' * 50}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate soros paper trading run.")
    parser.add_argument("--db", default=None, help="Path to soros.db (overrides DB_PATH env)")
    args = parser.parse_args()

    report = validate(db_path=args.db)
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
