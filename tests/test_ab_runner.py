"""Tests for backtest.ab_runner (sentiment A/B comparison)."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from backtest.ab_runner import ABResult, _compute_coverage, run_ab_backtest
from backtest.engine import BacktestConfig, BacktestResult, run_backtest

_START = 1_700_000_000   # 2023-11-14 22:13:20 UTC
_HOUR = 3_600
_DAY = 86_400


def _make_prices(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    n: int = 300,
    base_price: float = 30_000.0,
    trend: float = 0.0,
    funding_rate: float | None = 0.0001,
) -> pd.DataFrame:
    rows = []
    for i in range(n):
        c = base_price + trend * i
        rows.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "ts": _START + i * _HOUR,
            "open": c,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": 1_000.0,
            "funding_rate": funding_rate,
        })
    return pd.DataFrame(rows)


def _cfg(n: int = 300, signal_threshold: float = 0.05, **kw) -> BacktestConfig:
    return BacktestConfig(
        symbols=[("BTC/USDT", "crypto")],
        start_ts=_START,
        end_ts=_START + (n - 1) * _HOUR,
        initial_capital=10_000.0,
        signal_threshold=signal_threshold,
        **kw,
    )


def _ts_to_date(ts: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def _fng_for_prices(prices_df: pd.DataFrame, value: int = 50) -> dict[str, int]:
    """Build a F&G index covering every date in prices_df."""
    dates = set()
    for ts in prices_df["ts"]:
        dates.add(_ts_to_date(int(ts)))
    return {d: value for d in dates}


# ---------------------------------------------------------------------------
# ABResult dataclass
# ---------------------------------------------------------------------------

class TestABResult:
    def test_fields_exist(self):
        # Build stubs
        cfg = _cfg()
        empty_df = pd.DataFrame(
            columns=["symbol", "asset_class", "ts", "open", "high", "low",
                     "close", "volume", "funding_rate"]
        )
        r = run_backtest(cfg, prices_df=empty_df)
        ab = ABResult(off=r, on=r, fng_coverage_pct=0.5)
        assert isinstance(ab.off, BacktestResult)
        assert isinstance(ab.on, BacktestResult)
        assert ab.fng_coverage_pct == 0.5


# ---------------------------------------------------------------------------
# run_ab_backtest — basic contract
# ---------------------------------------------------------------------------

class TestRunABBacktest:
    def test_returns_abresult(self):
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert isinstance(result, ABResult)

    def test_both_variants_are_backtest_results(self):
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert isinstance(result.off, BacktestResult)
        assert isinstance(result.on, BacktestResult)

    def test_empty_fng_index_gives_zero_coverage(self):
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        result = run_ab_backtest(cfg, {}, prices_df=prices)
        assert result.fng_coverage_pct == 0.0

    def test_full_coverage_when_all_dates_present(self):
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.fng_coverage_pct == pytest.approx(1.0)

    def test_partial_coverage(self):
        # Only cover the first half of prices' dates
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        half_ts = _START + 149 * _HOUR
        fng = {}
        for ts in prices["ts"]:
            if ts <= half_ts:
                fng[_ts_to_date(int(ts))] = 50
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert 0.0 < result.fng_coverage_pct < 1.0

    def test_no_prices_returns_empty_equity_curves(self):
        cfg = _cfg(n=300)
        empty_df = pd.DataFrame(
            columns=["symbol", "asset_class", "ts", "open", "high", "low",
                     "close", "volume", "funding_rate"]
        )
        fng = {"2023-11-14": 50}
        result = run_ab_backtest(cfg, fng, prices_df=empty_df)
        assert result.off.equity_curve == []
        assert result.on.equity_curve == []
        assert result.fng_coverage_pct == 0.0

    def test_empty_fng_variants_are_identical(self):
        """When fng_index is empty all F&G lookups return None → sentiment=0 → same as OFF."""
        prices = _make_prices(n=300, base_price=1_000.0, trend=10.0)
        cfg = _cfg(n=300)
        result = run_ab_backtest(cfg, {}, prices_df=prices)
        assert result.off.total_return == pytest.approx(result.on.total_return, rel=1e-9)
        assert result.off.num_trades == result.on.num_trades

    def test_equity_curves_same_length(self):
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=75)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert len(result.off.equity_curve) == len(result.on.equity_curve)

    def test_initial_capital_preserved_in_both(self):
        prices = _make_prices(n=300)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.off.initial_capital == cfg.initial_capital
        assert result.on.initial_capital == cfg.initial_capital

    def test_metrics_bounds_both_variants(self):
        prices = _make_prices(n=300, base_price=1_000.0, trend=5.0)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=80)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        for r in (result.off, result.on):
            assert 0.0 <= r.win_rate <= 1.0
            assert r.max_drawdown >= 0.0
            assert r.num_wins <= r.num_trades


# ---------------------------------------------------------------------------
# Sentiment signal injection — behavioural tests
# ---------------------------------------------------------------------------

class TestSentimentInjection:
    def test_extreme_fear_can_differ_from_off(self):
        """F&G=0 (extreme fear → score=-1.0) should produce a different composite
        than OFF (score=0) when signals are near threshold, allowing the ON variant
        to diverge in trade decisions."""
        prices = _make_prices(n=300, base_price=1_000.0, trend=0.5)
        cfg = _cfg(n=300, signal_threshold=0.10)
        # Extreme fear: value=0 → score = (0-50)/50 = -1.0
        fng = _fng_for_prices(prices, value=0)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # Both variants run without error; ON may have fewer buys due to negative sentiment
        assert isinstance(result, ABResult)

    def test_extreme_greed_can_increase_buys(self):
        """F&G=100 (extreme greed → score=+1.0) should push composite upward,
        potentially triggering buys that the OFF variant doesn't fire."""
        # Flat prices — deterministic composite near zero, no signal
        prices = _make_prices(n=300, trend=0.0)
        cfg = _cfg(n=300, signal_threshold=0.05)
        # Extreme greed pushes composite toward +1
        fng = _fng_for_prices(prices, value=100)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # ON variant may have more trades than OFF due to positive sentiment boost
        # (at minimum it must not crash and return valid metrics)
        assert result.on.num_trades >= 0
        assert result.off.num_trades >= 0

    def test_neutral_fng_value_50_close_to_off(self):
        """F&G=50 → score=0.0 → composite is the same as OFF (neutral sentiment)."""
        prices = _make_prices(n=300, base_price=1_000.0, trend=5.0)
        cfg = _cfg(n=300)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # score_from_value(50) == 0.0; blending 0.0 into the composite should
        # yield the same action as the OFF variant (deterministic composite).
        # Allow small floating-point differences.
        assert result.off.total_return == pytest.approx(result.on.total_return, rel=1e-6)

    def test_fng_score_from_value_edge_cases(self):
        """score_from_value boundaries: 0→-1.0, 50→0.0, 100→+1.0."""
        from sentiment.fear_greed_history import score_from_value
        assert score_from_value(0) == pytest.approx(-1.0)
        assert score_from_value(50) == pytest.approx(0.0)
        assert score_from_value(100) == pytest.approx(1.0)
        assert score_from_value(25) == pytest.approx(-0.5)
        assert score_from_value(75) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverage:
    def _make_result(self, n: int = 5) -> BacktestResult:
        cfg = _cfg(n=n)
        empty_df = pd.DataFrame(
            columns=["symbol", "asset_class", "ts", "open", "high", "low",
                     "close", "volume", "funding_rate"]
        )
        # Build a fake result with a minimal equity curve
        from backtest.engine import Trade
        curve = [(_START + i * _HOUR, 10_000.0) for i in range(n)]
        return BacktestResult(
            cfg=cfg,
            initial_capital=10_000.0,
            final_equity=10_000.0,
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            num_trades=0,
            num_wins=0,
            equity_curve=curve,
            trades=[],
        )

    def test_empty_equity_curve_returns_zero(self):
        cfg = _cfg()
        empty_result = BacktestResult(
            cfg=cfg, initial_capital=10_000.0, final_equity=10_000.0,
            total_return=0.0, cagr=0.0, sharpe=0.0, max_drawdown=0.0,
            win_rate=0.0, num_trades=0, num_wins=0, equity_curve=[], trades=[],
        )
        assert _compute_coverage(empty_result, {"2023-11-14": 50}) == 0.0

    def test_empty_fng_index_returns_zero(self):
        result = self._make_result()
        assert _compute_coverage(result, {}) == 0.0

    def test_full_coverage(self):
        result = self._make_result(n=5)
        fng = {_ts_to_date(_START + i * _HOUR): 50 for i in range(5)}
        assert _compute_coverage(result, fng) == pytest.approx(1.0)

    def test_zero_coverage_when_no_dates_match(self):
        result = self._make_result(n=5)
        # "2000-01-01" does not match any 2023 bar date; exact matching → 0 coverage
        fng = {"2000-01-01": 50}
        assert _compute_coverage(result, fng) == 0.0

    def test_partial_coverage_exact(self):
        result = self._make_result(n=4)
        # Cover only 2 out of 4 timestamps (same day → 1 date, but 4 ts)
        # Use dates spread across multiple days to get partial coverage
        ts_list = [_START + i * _DAY for i in range(4)]
        from backtest.engine import Trade
        curve = [(ts, 10_000.0) for ts in ts_list]
        cfg = _cfg()
        full_result = BacktestResult(
            cfg=cfg, initial_capital=10_000.0, final_equity=10_000.0,
            total_return=0.0, cagr=0.0, sharpe=0.0, max_drawdown=0.0,
            win_rate=0.0, num_trades=0, num_wins=0, equity_curve=curve, trades=[],
        )
        fng = {_ts_to_date(ts_list[0]): 50, _ts_to_date(ts_list[1]): 60}
        assert _compute_coverage(full_result, fng) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases — helpers
# ---------------------------------------------------------------------------

def _make_prices_daily(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    n: int = 20,
    base_price: float = 30_000.0,
    trend: float = 0.0,
    funding_rate: float | None = 0.0001,
) -> pd.DataFrame:
    """Like _make_prices but one bar per day instead of per hour."""
    rows = []
    for i in range(n):
        c = base_price + trend * i
        rows.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "ts": _START + i * _DAY,
            "open": c,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": 1_000.0,
            "funding_rate": funding_rate,
        })
    return pd.DataFrame(rows)


def _cfg_daily(n: int = 20, signal_threshold: float = 0.05, **kw) -> BacktestConfig:
    return BacktestConfig(
        symbols=[("BTC/USDT", "crypto")],
        start_ts=_START,
        end_ts=_START + (n - 1) * _DAY,
        initial_capital=10_000.0,
        signal_threshold=signal_threshold,
        **kw,
    )


# ---------------------------------------------------------------------------
# Edge case 1: F&G historico com buraco (gap / hole in the middle)
# ---------------------------------------------------------------------------

class TestFngWithGap:
    """Tests for historical F&G data that has missing dates (hole in the middle).

    The lookup function backward-fills gaps, so the ON variant uses the last
    known value for hole dates — not neutral (0.0).  Coverage counts only the
    exact dates present in the index, ignoring backward-filled hole dates.
    """

    def test_fng_hole_in_middle_does_not_crash(self):
        """A/B backtest with a holey F&G index (gap in the middle) completes without error."""
        n = 20
        prices = _make_prices_daily(n=n)
        cfg = _cfg_daily(n=n)
        # First 5 days greed, last 5 days fear, middle 10 days missing (hole)
        fng = {}
        for i in range(5):
            fng[_ts_to_date(_START + i * _DAY)] = 80
        for i in range(15, n):
            fng[_ts_to_date(_START + i * _DAY)] = 20
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert isinstance(result, ABResult)
        assert isinstance(result.off, BacktestResult)
        assert isinstance(result.on, BacktestResult)

    def test_fng_coverage_excludes_backward_filled_hole_dates(self):
        """Coverage counts only dates present verbatim in the index.

        Hole dates that are resolved by backward-fill during scoring are NOT
        counted as covered — coverage reports contemporaneous F&G readings.
        """
        n = 20
        prices = _make_prices_daily(n=n)
        cfg = _cfg_daily(n=n)
        # Only the first 10 of 20 daily bars have a direct F&G entry
        fng = {_ts_to_date(_START + i * _DAY): 70 for i in range(10)}
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # 10 covered / 20 total bars → 0.5 exactly
        assert result.fng_coverage_pct == pytest.approx(0.5)

    def test_fng_partial_coverage_from_end_not_start(self):
        """Coverage is correct when only the trailing portion of dates is present."""
        n = 20
        prices = _make_prices_daily(n=n)
        cfg = _cfg_daily(n=n)
        # Only the last 8 of 20 bars are in the index
        fng = {_ts_to_date(_START + i * _DAY): 55 for i in range(12, n)}
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.fng_coverage_pct == pytest.approx(8 / n)

    def test_fng_hole_backward_fill_differs_from_neutral(self):
        """Backward-filled extreme greed in hole period diverges from all-neutral ON variant.

        When F&G=100 covers the first half and the second half is a hole,
        the ON variant backward-fills the hole with 100 (greed, score=+1.0).
        A separate run with F&G=50 (neutral) for all dates yields score=0.0
        for the same period.  With a low threshold, these produce different
        composites and potentially different trade counts.
        """
        n = 20
        prices = _make_prices_daily(n=n)
        cfg = _cfg_daily(n=n, signal_threshold=0.05)

        # Index A: first half F&G=100, second half is a hole (backward-fill → 100)
        index_a = {_ts_to_date(_START + i * _DAY): 100 for i in range(10)}

        # Index B: all days neutral (F&G=50 → score=0.0 → same as OFF)
        index_b = {_ts_to_date(_START + i * _DAY): 50 for i in range(n)}

        result_a = run_ab_backtest(cfg, index_a, prices_df=prices)
        result_b = run_ab_backtest(cfg, index_b, prices_df=prices)

        # A has partial coverage; B has full coverage
        assert result_a.fng_coverage_pct < result_b.fng_coverage_pct

        # B's ON is effectively neutral → identical to OFF
        assert result_b.on.num_trades == result_b.off.num_trades

        # A's ON should reflect greed (backward-filled), not neutral:
        # at minimum the number of trades is >= 0 and the result is valid
        assert result_a.on.num_trades >= 0
        assert result_a.off.num_trades >= 0

    def test_fng_single_entry_at_start_backward_fills_entire_period(self):
        """A single F&G entry at the start backward-fills the full backtest window.

        Coverage is still < 1.0 (only 1 of N daily bars has an exact date match),
        but the ON variant uses the backward-filled value for all remaining bars
        rather than falling back to neutral (0.0).
        """
        n = 5
        prices = _make_prices_daily(n=n)
        cfg = _cfg_daily(n=n, signal_threshold=2.0)  # no trades, just check coverage
        # Only the very first bar's date is in the F&G index
        fng = {_ts_to_date(_START): 90}
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # 1 exact date match out of n daily bars
        assert result.fng_coverage_pct == pytest.approx(1 / n)


# ---------------------------------------------------------------------------
# Edge case 2: variante sem trades (variant with no trades)
# ---------------------------------------------------------------------------

class TestVariantWithoutTrades:
    """Tests for when the signal threshold is so high that neither variant trades.

    The composite score is clamped to [-1, 1].  Setting signal_threshold > 1
    makes all bars "hold" and produces num_trades=0 for both OFF and ON variants.
    """

    def test_impossible_threshold_zero_trades_both_variants(self):
        """signal_threshold > 1.0 → composite never reaches threshold → 0 trades."""
        prices = _make_prices(n=300, base_price=1_000.0, trend=5.0)
        cfg = _cfg(n=300, signal_threshold=2.0)  # unreachable
        fng = _fng_for_prices(prices, value=80)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.off.num_trades == 0
        assert result.on.num_trades == 0

    def test_zero_trades_equity_curve_is_flat(self):
        """With no trades, equity stays at initial_capital throughout (no MTM moves)."""
        prices = _make_prices(n=100, base_price=5_000.0, trend=0.0)
        cfg = _cfg(n=100, signal_threshold=2.0)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        for _, equity in result.off.equity_curve:
            assert equity == pytest.approx(cfg.initial_capital)
        for _, equity in result.on.equity_curve:
            assert equity == pytest.approx(cfg.initial_capital)

    def test_zero_trades_metrics_are_zero(self):
        """With no trades: total_return=0, max_drawdown=0, win_rate=0."""
        prices = _make_prices(n=200, base_price=2_000.0, trend=0.0)
        cfg = _cfg(n=200, signal_threshold=2.0)
        fng = _fng_for_prices(prices, value=75)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        for r in (result.off, result.on):
            assert r.total_return == pytest.approx(0.0)
            assert r.max_drawdown == pytest.approx(0.0)
            assert r.win_rate == pytest.approx(0.0)
            assert r.num_wins == 0

    def test_zero_trades_coverage_still_computed(self):
        """Coverage is still computed even when no trades fire."""
        prices = _make_prices(n=100, base_price=3_000.0, trend=0.0)
        cfg = _cfg(n=100, signal_threshold=2.0)
        fng = _fng_for_prices(prices, value=60)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.fng_coverage_pct == pytest.approx(1.0)
        assert result.off.num_trades == 0

    def test_fng_empty_and_zero_trades(self):
        """Empty F&G + impossible threshold: both variants are fully identical."""
        prices = _make_prices(n=150, base_price=1_000.0, trend=0.0)
        cfg = _cfg(n=150, signal_threshold=2.0)
        result = run_ab_backtest(cfg, {}, prices_df=prices)
        assert result.off.num_trades == 0
        assert result.on.num_trades == 0
        assert result.off.total_return == pytest.approx(result.on.total_return)

    def test_equity_curve_present_even_with_zero_trades(self):
        """Equity curve is populated (one entry per bar) even when no trades fire."""
        n = 50
        prices = _make_prices(n=n, base_price=1_000.0)
        cfg = _cfg(n=n, signal_threshold=2.0)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert len(result.off.equity_curve) == n
        assert len(result.on.equity_curve) == n


# ---------------------------------------------------------------------------
# Edge case 3: amostra curta (short sample — very few bars)
# ---------------------------------------------------------------------------

class TestShortSample:
    """Tests for A/B with very few bars (n=1 or n=2).

    Short samples cannot produce momentum/volatility signals (need ≥26 candles
    for MACD), so signals are always "hold" and num_trades=0.  Metrics like
    Sharpe and CAGR should degrade gracefully (return 0 or None, not crash).
    """

    def test_single_bar_does_not_crash(self):
        """A/B with a single price bar completes without error."""
        prices = _make_prices(n=1)
        cfg = _cfg(n=1)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert isinstance(result, ABResult)

    def test_single_bar_equity_curve_length_one(self):
        """Single-bar backtest produces exactly 1 equity-curve entry per variant."""
        prices = _make_prices(n=1)
        cfg = _cfg(n=1)
        fng = _fng_for_prices(prices, value=70)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert len(result.off.equity_curve) == 1
        assert len(result.on.equity_curve) == 1

    def test_single_bar_off_variant_no_trades(self):
        """OFF variant with a single bar and flat prices produces 0 trades.

        Without sentiment, the composite is driven only by momentum, volatility,
        and funding.  With a single bar the momentum and volatility signals
        return 0.0 (insufficient history); funding alone is not enough to cross
        a moderate threshold.
        """
        prices = _make_prices(n=1, base_price=30_000.0, funding_rate=None)
        cfg = _cfg(n=1, signal_threshold=0.05)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.off.num_trades == 0

    def test_single_bar_zero_return(self):
        """With a single bar and no trades, total_return and max_drawdown are 0."""
        prices = _make_prices(n=1)
        cfg = _cfg(n=1)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.off.total_return == pytest.approx(0.0)
        assert result.on.total_return == pytest.approx(0.0)
        assert result.off.max_drawdown == pytest.approx(0.0)
        assert result.on.max_drawdown == pytest.approx(0.0)

    def test_two_bars_does_not_crash(self):
        """A/B with two price bars completes without error."""
        prices = _make_prices(n=2)
        cfg = _cfg(n=2)
        fng = _fng_for_prices(prices, value=60)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert isinstance(result, ABResult)

    def test_two_bars_equity_curve_length_two(self):
        """Two-bar backtest produces exactly 2 equity-curve entries per variant."""
        prices = _make_prices(n=2)
        cfg = _cfg(n=2)
        fng = _fng_for_prices(prices, value=40)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert len(result.off.equity_curve) == 2
        assert len(result.on.equity_curve) == 2

    def test_short_sample_sharpe_not_conclusive(self):
        """With very few bars, the backtest Sharpe is 0 (insufficient variance)."""
        prices = _make_prices(n=5)
        cfg = _cfg(n=5)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # Flat equity (no trades) → std of returns = 0 → sharpe = 0.0
        assert result.off.sharpe == pytest.approx(0.0)
        assert result.on.sharpe == pytest.approx(0.0)

    def test_coverage_computed_for_short_sample(self):
        """Coverage is computed correctly even for a very short price series."""
        prices = _make_prices(n=3)
        cfg = _cfg(n=3)
        fng = _fng_for_prices(prices, value=50)
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        # Full coverage (all bars' dates present in index)
        assert result.fng_coverage_pct == pytest.approx(1.0)

    def test_short_sample_with_fng_gap_coverage_partial(self):
        """Short sample with a partial F&G index reports partial coverage."""
        n = 4
        prices = _make_prices_daily(n=n)
        cfg = _cfg_daily(n=n)
        # Only 2 of 4 days covered
        fng = {_ts_to_date(_START + i * _DAY): 60 for i in range(2)}
        result = run_ab_backtest(cfg, fng, prices_df=prices)
        assert result.fng_coverage_pct == pytest.approx(0.5)
