"""Backtest A/B runner: sentiment OFF vs ON using Fear & Greed history.

Runs the backtest engine twice over the same price window — once with
sentiment disabled (deterministic signals only) and once with the historical
Fear & Greed index injected as the sentiment signal — and returns both
BacktestResult instances plus a F&G coverage metric.

Public API
----------
ABResult            dataclass holding the two BacktestResult objects
run_ab_backtest     main entry point: runs both variants and returns ABResult

Usage
-----
    from backtest.ab_runner import run_ab_backtest
    from backtest.engine import BacktestConfig
    from sentiment.fear_greed_history import get_index

    fng_index = get_index(cache_path="/tmp/fng_cache.json")
    cfg = BacktestConfig(
        symbols=[("BTC/USDT", "crypto")],
        start_ts=1_700_000_000,
        end_ts=1_710_000_000,
    )
    result = run_ab_backtest(cfg, fng_index)
    print(result.off.total_return, result.on.total_return, result.fng_coverage_pct)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestConfig, BacktestResult, run_backtest
from sentiment.fear_greed_history import lookup, score_from_value


@dataclass
class ABResult:
    """Sentiment A/B comparison result.

    off: backtest without sentiment (deterministic composite only)
    on:  backtest with F&G sentiment injected per bar
    fng_coverage_pct: fraction of bars in the backtest range that had a
        non-None F&G value (0.0 when fng_index is empty, 1.0 for full coverage)
    """
    off: BacktestResult
    on: BacktestResult
    fng_coverage_pct: float


def run_ab_backtest(
    cfg: BacktestConfig,
    fng_index: dict[str, int],
    *,
    prices_df: pd.DataFrame | None = None,
) -> ABResult:
    """Run the sentiment A/B backtest comparison.

    Runs ``run_backtest`` twice with the same *cfg* and price data:

    * **OFF** variant: no sentiment (deterministic signals only, existing behaviour).
    * **ON** variant: F&G value at each bar is converted to a score via
      ``score_from_value`` and blended into the composite using the full
      class-weighted formula (same weights as the live aggregator).

    When a bar's date precedes the earliest record in *fng_index*, the
    sentiment score for that bar defaults to 0.0 (neutral), preserving the
    same composite the OFF variant would produce at that bar.

    Args:
        cfg: Backtest configuration shared by both runs.
        fng_index: Date-indexed F&G values built by
            ``sentiment.fear_greed_history.build_index``.  Pass an empty dict
            to get identical OFF/ON results (coverage = 0 %).
        prices_df: Optional pre-loaded price DataFrame (same schema as
            ``run_backtest``).  When None, each variant loads prices from the
            configured SQLite DB independently.

    Returns:
        ABResult with both BacktestResult objects and the F&G coverage fraction.
    """
    result_off = run_backtest(cfg, prices_df=prices_df)

    fng_coverage_pct = _compute_coverage(result_off, fng_index)

    def _sentiment_fn(ts: int, symbol: str, asset_class: str) -> float:
        value = lookup(fng_index, ts)
        if value is None:
            return 0.0
        return score_from_value(value)

    result_on = run_backtest(cfg, prices_df=prices_df, sentiment_fn=_sentiment_fn)

    return ABResult(
        off=result_off,
        on=result_on,
        fng_coverage_pct=fng_coverage_pct,
    )


def _compute_coverage(result_off: BacktestResult, fng_index: dict[str, int]) -> float:
    """Return the fraction of equity-curve timestamps whose calendar date is in *fng_index*.

    Uses exact date matching (no backward fill) so the metric tells users how much of
    the backtest range had contemporaneous F&G readings rather than filled-forward values.
    """
    if not result_off.equity_curve or not fng_index:
        return 0.0
    timestamps = [ts for ts, _ in result_off.equity_curve]
    covered = sum(
        1 for ts in timestamps
        if datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d") in fng_index
    )
    return covered / len(timestamps)
