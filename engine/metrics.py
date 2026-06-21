"""Comparative performance metrics for soros vs buy-and-hold BTC benchmark.

Pure computation layer — no I/O, no side effects.

Public API:
  - ComparisonMetrics: frozen dataclass holding all metrics for both series
  - compute_metrics: derive ComparisonMetrics from a BenchmarkSeries
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from engine.benchmark import BenchmarkSeries

# Sharpe is noisy below this many periods; we report it but flag as inconclusive.
MIN_SHARPE_N = 30

# Seconds per year for annualisation denominator (non-leap average).
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class ComparisonMetrics:
    """Performance metrics for both equity curves derived from a BenchmarkSeries.

    All return/drawdown values are expressed as decimals (0.15 = +15%).
    Sharpe is annualised; risk_free_rate is always 0.0 for crypto.
    """

    # --- soros ---
    soros_total_return: float       # (final - initial) / initial
    soros_sharpe: float | None      # None when std == 0 or n < 2; see sharpe_conclusive
    soros_max_drawdown: float       # ≤ 0; worst peak-to-trough as a fraction

    # --- BTC buy-and-hold ---
    btc_total_return: float
    btc_sharpe: float | None
    btc_max_drawdown: float

    # --- shared metadata ---
    n: int                          # number of aligned data points
    annualization_factor: float     # periods per year used for Sharpe scaling
    median_interval_seconds: float  # detected cadence between snapshots
    sharpe_conclusive: bool         # True only when n >= MIN_SHARPE_N
    risk_free_rate: float           # 0.0 (declared; always zero for crypto)


def compute_metrics(series: BenchmarkSeries) -> ComparisonMetrics:
    """Derive ComparisonMetrics from an aligned BenchmarkSeries.

    Annualisation factor is inferred from the median interval between
    consecutive snapshot timestamps so that hourly and daily series both
    produce a sensible annualised Sharpe.

    A Sharpe is always computed (when mathematically possible) but
    ``sharpe_conclusive`` is False when n < MIN_SHARPE_N, signalling
    that the estimate is unreliable with so few data points.

    Args:
        series: BenchmarkSeries produced by build_btc_benchmark.

    Returns:
        ComparisonMetrics with metrics for both curves.

    Raises:
        ValueError: if series has fewer than 1 data point.
    """
    n = series.n_points
    if n < 1:
        raise ValueError("series has no data points")

    soros = series.soros_equity
    btc = series.btc_equity
    ts = series.timestamps

    soros_total_return = _total_return(soros)
    btc_total_return = _total_return(btc)

    soros_max_drawdown = _max_drawdown(soros)
    btc_max_drawdown = _max_drawdown(btc)

    median_interval = _median_interval(ts)
    ann_factor = _annualization_factor(median_interval)
    conclusive = n >= MIN_SHARPE_N

    soros_sharpe = _sharpe(soros, ann_factor)
    btc_sharpe = _sharpe(btc, ann_factor)

    return ComparisonMetrics(
        soros_total_return=soros_total_return,
        soros_sharpe=soros_sharpe,
        soros_max_drawdown=soros_max_drawdown,
        btc_total_return=btc_total_return,
        btc_sharpe=btc_sharpe,
        btc_max_drawdown=btc_max_drawdown,
        n=n,
        annualization_factor=ann_factor,
        median_interval_seconds=median_interval,
        sharpe_conclusive=conclusive,
        risk_free_rate=0.0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _total_return(equity: tuple[float, ...]) -> float:
    if len(equity) < 1:
        return 0.0
    return (equity[-1] - equity[0]) / equity[0]


def _max_drawdown(equity: tuple[float, ...]) -> float:
    """Maximum peak-to-trough drawdown as a fraction (≤ 0)."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _period_returns(equity: tuple[float, ...]) -> list[float]:
    return [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, len(equity))]


def _sharpe(equity: tuple[float, ...], ann_factor: float) -> float | None:
    """Annualised Sharpe with risk-free rate = 0."""
    if len(equity) < 2:
        return None
    rets = _period_returns(equity)
    if len(rets) < 1:
        return None
    try:
        std = statistics.stdev(rets)
    except statistics.StatisticsError:
        return None
    if std == 0.0:
        return None
    mean = statistics.mean(rets)
    return mean / std * math.sqrt(ann_factor)


def _median_interval(timestamps: tuple[int, ...]) -> float:
    """Median gap in seconds between consecutive timestamps.

    Returns 3600.0 (hourly) as a safe default for single-point series.
    """
    if len(timestamps) < 2:
        return 3600.0
    diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    return float(statistics.median(diffs))


def _annualization_factor(median_interval_seconds: float) -> float:
    """Periods per year given a median snapshot interval in seconds."""
    if median_interval_seconds <= 0:
        return _SECONDS_PER_YEAR / 3600.0  # fallback: hourly
    return _SECONDS_PER_YEAR / median_interval_seconds
