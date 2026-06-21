"""Buy-and-hold BTC benchmark construction.

Pure computation layer:
  - BenchmarkSeries: frozen dataclass holding both aligned equity curves
  - build_btc_benchmark: pure function; no I/O

DB helpers (thin, read-only, accept a Connection for testability):
  - load_equity_snapshots: reads equity_curve table
  - load_btc_closes: reads prices table for BTC/USDT
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BenchmarkSeries:
    """Aligned equity curves for soros vs buy-and-hold BTC.

    Both series share the same initial capital and the same window start.
    If BTC data starts after the first equity snapshot, the common window
    begins at the first snapshot that has BTC coverage.
    """

    timestamps: tuple[int, ...]       # unix seconds, ascending
    soros_equity: tuple[float, ...]   # equity_curve values at each timestamp
    btc_equity: tuple[float, ...]     # buy-and-hold BTC equity, same initial capital
    initial_capital: float            # == soros_equity[0]; starting value for both curves
    btc_start_price: float            # BTC/USDT close used as the buy-and-hold reference
    window_start: int                 # == timestamps[0]
    window_end: int                   # == timestamps[-1]
    n_points: int                     # == len(timestamps)
    n_btc_gaps: int                   # snapshots where BTC price was forward-filled


def build_btc_benchmark(
    snapshots: Sequence[tuple[int, float]],
    btc_closes: Sequence[tuple[int, float]],
) -> BenchmarkSeries:
    """Build a buy-and-hold BTC equity curve aligned to soros equity snapshots.

    For each equity snapshot timestamp, locates the most recent BTC/USDT close
    at or before that timestamp (forward-fill for gaps).  Snapshots that fall
    before the first available BTC price are skipped and excluded from the
    aligned window.

    Args:
        snapshots: (unix_ts, equity) pairs from equity_curve, any order.
        btc_closes: (unix_ts, close) pairs for BTC/USDT, any order.

    Returns:
        BenchmarkSeries with both curves normalised to the same initial capital
        at the common window start.

    Raises:
        ValueError: if snapshots is empty, or no BTC close overlaps the window.
    """
    if not snapshots:
        raise ValueError("snapshots is empty")

    snap = sorted(snapshots, key=lambda x: x[0])

    # Build sorted parallel arrays for BTC — bisect needs a list of keys
    sorted_btc = sorted(btc_closes, key=lambda x: x[0])
    btc_ts_arr: list[int] = [ts for ts, _ in sorted_btc]
    btc_price_arr: list[float] = [price for _, price in sorted_btc]

    out_timestamps: list[int] = []
    out_soros: list[float] = []
    out_btc_raw: list[float] = []
    n_btc_gaps = 0
    btc_start_price: float | None = None

    for snap_ts, snap_eq in snap:
        idx = bisect.bisect_right(btc_ts_arr, snap_ts) - 1
        if idx < 0:
            # No BTC price at or before this snapshot — outside common window
            continue

        btc_price = btc_price_arr[idx]
        if btc_ts_arr[idx] != snap_ts:
            n_btc_gaps += 1

        if btc_start_price is None:
            btc_start_price = btc_price

        out_timestamps.append(snap_ts)
        out_soros.append(snap_eq)
        out_btc_raw.append(btc_price)

    if btc_start_price is None:
        raise ValueError("no BTC close prices overlap with the equity snapshot window")

    initial_capital = out_soros[0]
    btc_eq = tuple(initial_capital * (p / btc_start_price) for p in out_btc_raw)

    return BenchmarkSeries(
        timestamps=tuple(out_timestamps),
        soros_equity=tuple(out_soros),
        btc_equity=btc_eq,
        initial_capital=initial_capital,
        btc_start_price=btc_start_price,
        window_start=out_timestamps[0],
        window_end=out_timestamps[-1],
        n_points=len(out_timestamps),
        n_btc_gaps=n_btc_gaps,
    )


# ---------------------------------------------------------------------------
# DB helpers — thin read-only wrappers; pure functions call build_btc_benchmark
# ---------------------------------------------------------------------------

def load_equity_snapshots(
    conn: sqlite3.Connection,
    *,
    is_paper: bool | None = None,
) -> list[tuple[int, float]]:
    """Return (ts, equity) rows from equity_curve, ordered by ts ascending."""
    if is_paper is None:
        rows = conn.execute(
            "SELECT ts, equity FROM equity_curve ORDER BY ts ASC, id ASC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts, equity FROM equity_curve"
            " WHERE is_paper = ? ORDER BY ts ASC, id ASC",
            (int(is_paper),),
        ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def load_btc_closes(
    conn: sqlite3.Connection,
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: int | None = None,
    until: int | None = None,
) -> list[tuple[int, float]]:
    """Return (ts, close) rows for BTC/USDT from prices, ordered by ts ascending."""
    params: list = [symbol, timeframe]
    where = "symbol = ? AND timeframe = ?"
    if since is not None:
        where += " AND ts >= ?"
        params.append(since)
    if until is not None:
        where += " AND ts <= ?"
        params.append(until)
    rows = conn.execute(
        f"SELECT ts, close FROM prices WHERE {where} ORDER BY ts ASC",  # noqa: S608
        params,
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]
