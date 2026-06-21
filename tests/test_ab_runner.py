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
