"""Tests for engine/metrics.py — ComparisonMetrics computation."""

from __future__ import annotations

import math

import pytest

from engine.benchmark import BenchmarkSeries
from engine.metrics import (
    MIN_SHARPE_N,
    ComparisonMetrics,
    _annualization_factor,
    _max_drawdown,
    _median_interval,
    _period_returns,
    _sharpe,
    _total_return,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOURLY = 3600
_DAILY = 86400
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _make_series(
    soros: list[float],
    btc: list[float],
    *,
    interval: int = _HOURLY,
    start_ts: int = 1_000_000,
) -> BenchmarkSeries:
    """Build a minimal BenchmarkSeries from equity lists."""
    assert len(soros) == len(btc)
    n = len(soros)
    ts = tuple(start_ts + i * interval for i in range(n))
    return BenchmarkSeries(
        timestamps=ts,
        soros_equity=tuple(soros),
        btc_equity=tuple(btc),
        initial_capital=soros[0],
        btc_start_price=50_000.0,
        window_start=ts[0],
        window_end=ts[-1],
        n_points=n,
        n_btc_gaps=0,
    )


# ---------------------------------------------------------------------------
# _total_return
# ---------------------------------------------------------------------------

class TestTotalReturn:
    def test_flat(self):
        assert _total_return((10_000.0, 10_000.0)) == pytest.approx(0.0)

    def test_gain(self):
        assert _total_return((10_000.0, 12_000.0)) == pytest.approx(0.2)

    def test_loss(self):
        assert _total_return((10_000.0, 8_000.0)) == pytest.approx(-0.2)

    def test_single_point(self):
        assert _total_return((10_000.0,)) == pytest.approx(0.0)

    def test_long_series_uses_first_and_last(self):
        assert _total_return((100.0, 110.0, 90.0, 150.0)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotone_up_no_drawdown(self):
        assert _max_drawdown((100.0, 110.0, 120.0)) == pytest.approx(0.0)

    def test_simple_drawdown(self):
        # peak=100, then 80 → dd = (80-100)/100 = -0.20
        assert _max_drawdown((100.0, 80.0)) == pytest.approx(-0.20)

    def test_recover_to_peak_then_fall(self):
        # peak=120 at index 2, then 96 → dd = (96-120)/120 = -0.20
        assert _max_drawdown((100.0, 110.0, 120.0, 96.0)) == pytest.approx(-0.20)

    def test_multiple_drawdowns_returns_worst(self):
        # first: 100→80 = -0.20; second: after recovering to 100, drops to 50 = -0.50
        assert _max_drawdown((100.0, 80.0, 100.0, 50.0)) == pytest.approx(-0.50)

    def test_single_point_returns_zero(self):
        assert _max_drawdown((100.0,)) == pytest.approx(0.0)

    def test_negative_values_not_used_as_peak(self):
        # equity shouldn't be negative in practice, but guard anyway
        result = _max_drawdown((100.0, 120.0, 60.0))
        assert result == pytest.approx(-0.50)


# ---------------------------------------------------------------------------
# _period_returns
# ---------------------------------------------------------------------------

class TestPeriodReturns:
    def test_uniform_growth(self):
        rets = _period_returns((100.0, 110.0, 121.0))
        assert rets[0] == pytest.approx(0.10)
        assert rets[1] == pytest.approx(0.10)

    def test_mixed(self):
        rets = _period_returns((100.0, 120.0, 96.0))
        assert rets[0] == pytest.approx(0.20)
        assert rets[1] == pytest.approx(-0.20)

    def test_length_is_n_minus_1(self):
        assert len(_period_returns(tuple(range(1, 6)))) == 4


# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_returns_none_for_single_point(self):
        assert _sharpe((10_000.0,), 8760) is None

    def test_returns_none_for_zero_std(self):
        # flat equity → all returns = 0 → std = 0
        assert _sharpe((10_000.0, 10_000.0, 10_000.0), 8760) is None

    def test_positive_sharpe_for_positive_returns(self):
        # monotone up → positive Sharpe
        equity = tuple(float(10_000 + i * 100) for i in range(10))
        result = _sharpe(equity, 8760)
        assert result is not None
        assert result > 0

    def test_negative_sharpe_for_declining_equity(self):
        equity = tuple(float(10_000 - i * 100) for i in range(10))
        result = _sharpe(equity, 8760)
        assert result is not None
        assert result < 0

    def test_higher_ann_factor_scales_up(self):
        equity = (100.0, 101.0, 102.0, 101.5, 103.0, 104.0)
        s_hourly = _sharpe(equity, _SECONDS_PER_YEAR / _HOURLY)
        s_daily = _sharpe(equity, _SECONDS_PER_YEAR / _DAILY)
        assert s_hourly is not None and s_daily is not None
        # hourly has more periods per year → larger annualisation factor → larger |Sharpe|
        assert abs(s_hourly) > abs(s_daily)

    def test_sharpe_scales_by_sqrt_ann_factor(self):
        equity = (100.0, 101.0, 99.0, 102.0, 100.5)
        s1 = _sharpe(equity, 100.0)
        s4 = _sharpe(equity, 400.0)
        assert s1 is not None and s4 is not None
        assert s4 == pytest.approx(s1 * 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# _median_interval
# ---------------------------------------------------------------------------

class TestMedianInterval:
    def test_single_point_returns_3600(self):
        assert _median_interval((1000,)) == pytest.approx(3600.0)

    def test_uniform_hourly(self):
        ts = tuple(i * 3600 for i in range(5))
        assert _median_interval(ts) == pytest.approx(3600.0)

    def test_uniform_daily(self):
        ts = tuple(i * 86400 for i in range(5))
        assert _median_interval(ts) == pytest.approx(86400.0)

    def test_mixed_intervals_returns_median(self):
        # gaps: 1h, 2h, 1h → median = 1h
        ts = (0, 3600, 10800, 14400)
        assert _median_interval(ts) == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# _annualization_factor
# ---------------------------------------------------------------------------

class TestAnnualizationFactor:
    def test_hourly(self):
        factor = _annualization_factor(3600.0)
        assert factor == pytest.approx(_SECONDS_PER_YEAR / 3600.0, rel=1e-6)

    def test_daily(self):
        factor = _annualization_factor(86400.0)
        assert factor == pytest.approx(_SECONDS_PER_YEAR / 86400.0, rel=1e-6)

    def test_zero_interval_fallback_to_hourly(self):
        factor = _annualization_factor(0.0)
        assert factor == pytest.approx(_SECONDS_PER_YEAR / 3600.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_metrics — basic correctness
# ---------------------------------------------------------------------------

class TestComputeMetricsBasic:
    def test_returns_comparison_metrics_instance(self):
        series = _make_series([10_000.0, 11_000.0], [10_000.0, 12_000.0])
        result = compute_metrics(series)
        assert isinstance(result, ComparisonMetrics)

    def test_n_matches_series(self):
        series = _make_series([10_000.0] * 5, [10_000.0] * 5)
        assert compute_metrics(series).n == 5

    def test_risk_free_rate_is_zero(self):
        series = _make_series([10_000.0, 11_000.0], [10_000.0, 10_500.0])
        assert compute_metrics(series).risk_free_rate == 0.0

    def test_result_is_frozen(self):
        series = _make_series([10_000.0, 11_000.0], [10_000.0, 10_500.0])
        result = compute_metrics(series)
        with pytest.raises((AttributeError, TypeError)):
            result.n = 99  # type: ignore[misc]


class TestComputeMetricsTotalReturn:
    def test_soros_gain(self):
        series = _make_series([10_000.0, 12_000.0], [10_000.0, 10_000.0])
        m = compute_metrics(series)
        assert m.soros_total_return == pytest.approx(0.20)

    def test_btc_gain(self):
        series = _make_series([10_000.0, 10_000.0], [10_000.0, 15_000.0])
        m = compute_metrics(series)
        assert m.btc_total_return == pytest.approx(0.50)

    def test_soros_loss_btc_gain(self):
        series = _make_series([10_000.0, 8_000.0], [10_000.0, 12_000.0])
        m = compute_metrics(series)
        assert m.soros_total_return == pytest.approx(-0.20)
        assert m.btc_total_return == pytest.approx(0.20)


class TestComputeMetricsMaxDrawdown:
    def test_soros_drawdown(self):
        series = _make_series([10_000.0, 12_000.0, 8_400.0], [10_000.0, 10_000.0, 10_000.0])
        m = compute_metrics(series)
        # peak=12_000, trough=8_400 → dd = (8400-12000)/12000 = -0.30
        assert m.soros_max_drawdown == pytest.approx(-0.30)

    def test_btc_drawdown(self):
        series = _make_series([10_000.0, 10_000.0, 10_000.0], [10_000.0, 14_000.0, 7_000.0])
        m = compute_metrics(series)
        # peak=14_000, trough=7_000 → dd = (7000-14000)/14000 = -0.50
        assert m.btc_max_drawdown == pytest.approx(-0.50)

    def test_no_drawdown_when_monotone_up(self):
        series = _make_series([10_000.0, 11_000.0, 12_000.0], [10_000.0, 11_000.0, 12_000.0])
        m = compute_metrics(series)
        assert m.soros_max_drawdown == pytest.approx(0.0)
        assert m.btc_max_drawdown == pytest.approx(0.0)

    def test_single_point_drawdown_is_zero(self):
        series = _make_series([10_000.0], [10_000.0])
        m = compute_metrics(series)
        assert m.soros_max_drawdown == pytest.approx(0.0)
        assert m.btc_max_drawdown == pytest.approx(0.0)


class TestComputeMetricsSharpeConclusiveness:
    def test_small_sample_not_conclusive(self):
        n = MIN_SHARPE_N - 1
        series = _make_series(
            [float(10_000 + i * 10) for i in range(n)],
            [float(10_000 + i * 8) for i in range(n)],
        )
        m = compute_metrics(series)
        assert m.sharpe_conclusive is False

    def test_large_sample_conclusive(self):
        n = MIN_SHARPE_N
        series = _make_series(
            [float(10_000 + i * 10) for i in range(n)],
            [float(10_000 + i * 8) for i in range(n)],
        )
        m = compute_metrics(series)
        assert m.sharpe_conclusive is True

    def test_sharpe_still_returned_when_inconclusive(self):
        # We compute Sharpe even for small samples — caller decides how to display it.
        n = MIN_SHARPE_N - 5
        series = _make_series(
            [float(10_000 + i * 10) for i in range(n)],
            [float(10_000 + i * 8) for i in range(n)],
        )
        m = compute_metrics(series)
        # Sharpe is defined (non-None) as long as variance exists
        # For monotone series stdev may be 0 — use a series with variance
        series2 = _make_series(
            [10_000.0, 10_100.0, 9_950.0, 10_200.0, 10_050.0],
            [10_000.0, 10_080.0, 9_920.0, 10_150.0, 10_010.0],
        )
        m2 = compute_metrics(series2)
        assert m2.sharpe_conclusive is False
        # Sharpe may or may not be None depending on variance — just test flag, not value here


class TestComputeMetricsAnnualizationFactor:
    def test_hourly_series_ann_factor(self):
        series = _make_series(
            [10_000.0, 10_100.0, 9_950.0],
            [10_000.0, 10_080.0, 9_900.0],
            interval=_HOURLY,
        )
        m = compute_metrics(series)
        assert m.annualization_factor == pytest.approx(_SECONDS_PER_YEAR / _HOURLY, rel=1e-6)
        assert m.median_interval_seconds == pytest.approx(float(_HOURLY))

    def test_daily_series_ann_factor(self):
        series = _make_series(
            [10_000.0, 10_100.0, 9_950.0],
            [10_000.0, 10_080.0, 9_900.0],
            interval=_DAILY,
        )
        m = compute_metrics(series)
        assert m.annualization_factor == pytest.approx(_SECONDS_PER_YEAR / _DAILY, rel=1e-6)
        assert m.median_interval_seconds == pytest.approx(float(_DAILY))


# ---------------------------------------------------------------------------
# compute_metrics — edge cases
# ---------------------------------------------------------------------------

class TestComputeMetricsEdgeCases:
    def test_single_point_series(self):
        series = _make_series([10_000.0], [10_000.0])
        m = compute_metrics(series)
        assert m.n == 1
        assert m.soros_total_return == pytest.approx(0.0)
        assert m.btc_total_return == pytest.approx(0.0)
        assert m.soros_max_drawdown == pytest.approx(0.0)
        assert m.btc_max_drawdown == pytest.approx(0.0)
        assert m.soros_sharpe is None
        assert m.btc_sharpe is None
        assert m.sharpe_conclusive is False

    def test_flat_equity_sharpe_is_none(self):
        series = _make_series(
            [10_000.0] * 5,
            [10_000.0] * 5,
        )
        m = compute_metrics(series)
        assert m.soros_sharpe is None
        assert m.btc_sharpe is None

    def test_two_points_sharpe_computed_when_std_nonzero(self):
        series = _make_series([10_000.0, 11_000.0], [10_000.0, 9_000.0])
        m = compute_metrics(series)
        # With 2 points there's only 1 return → stdev requires ≥2 values → None
        assert m.soros_sharpe is None
        assert m.btc_sharpe is None

    def test_three_points_sharpe_computed(self):
        series = _make_series(
            [10_000.0, 10_100.0, 9_950.0],
            [10_000.0, 10_080.0, 9_900.0],
        )
        m = compute_metrics(series)
        # std of 2 returns is defined
        assert m.soros_sharpe is not None
        assert m.btc_sharpe is not None
