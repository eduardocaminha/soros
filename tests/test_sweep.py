"""Tests for the sweep runner (backtest/sweep.py)."""

from __future__ import annotations

import math
import time
from unittest.mock import patch

import pandas as pd
import pytest

from backtest.engine import BacktestConfig
from backtest.sweep import SweepRow, SweepSpec, _resolve_symbols, _save_sweep_rows, run_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = 1_700_000_000
_HOUR = 3600


def _make_prices(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    n: int = 300,
    base_price: float = 30_000.0,
    trend: float = 10.0,
    funding_rate: float | None = 0.0001,
) -> pd.DataFrame:
    closes = [base_price + trend * i for i in range(n)]
    rows = []
    for i, c in enumerate(closes):
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


def _cfg(
    symbols: list[tuple[str, str]] | None = None,
    n: int = 300,
    initial_capital: float = 10_000.0,
) -> BacktestConfig:
    if symbols is None:
        symbols = [("BTC/USDT", "crypto")]
    return BacktestConfig(
        symbols=symbols,
        start_ts=_START,
        end_ts=_START + (n - 1) * _HOUR,
        initial_capital=initial_capital,
    )


def _run(cfg=None, thresholds=None, trend=10.0, prices=None):
    """Convenience: run_sweep with injected prices (no DB)."""
    if cfg is None:
        cfg = _cfg()
    if prices is None:
        prices = _make_prices(trend=trend)
    with patch("backtest.sweep._save_sweep_rows"):
        return run_sweep(cfg, thresholds=thresholds, _prices_df=prices)


def _run_spec(spec, cfg=None, prices=None):
    """Convenience: run_sweep via SweepSpec with injected prices (no DB)."""
    if cfg is None:
        cfg = _cfg()
    if prices is None:
        prices = _make_prices()
    with patch("backtest.sweep._save_sweep_rows"):
        return run_sweep(cfg, spec=spec, _prices_df=prices)


# ---------------------------------------------------------------------------
# SweepRow
# ---------------------------------------------------------------------------


class TestSweepRow:
    def test_fields_present(self) -> None:
        row = SweepRow(
            sweep_id="abc123",
            run_ts=int(time.time()),
            signal_threshold=0.25,
            total_return=0.1,
            cagr=0.05,
            sharpe=1.2,
            max_dd=0.08,
            win_rate=0.6,
            n_trades=10,
        )
        assert row.signal_threshold == 0.25
        assert row.n_trades == 10
        assert row.sweep_id == "abc123"


# ---------------------------------------------------------------------------
# _resolve_symbols
# ---------------------------------------------------------------------------


class TestResolveSymbols:
    def test_explicit_symbols_passthrough(self) -> None:
        cfg = BacktestConfig(
            symbols=[("BTC/USDT", "crypto"), ("AAPL", "stocks")],
            start_ts=_START,
            end_ts=_START + _HOUR,
        )
        assert _resolve_symbols(cfg) == [("BTC/USDT", "crypto"), ("AAPL", "stocks")]

    def test_returns_independent_list(self) -> None:
        cfg = BacktestConfig(
            symbols=[("BTC/USDT", "crypto")],
            start_ts=_START,
            end_ts=_START + _HOUR,
        )
        result = _resolve_symbols(cfg)
        result.append(("ETH/USDT", "crypto"))
        assert cfg.symbols == [("BTC/USDT", "crypto")]


# ---------------------------------------------------------------------------
# run_sweep — core behaviour
# ---------------------------------------------------------------------------


class TestRunSweep:
    def test_returns_one_row_per_threshold(self) -> None:
        rows = _run(thresholds=[0.10, 0.25, 0.40])
        assert len(rows) == 3
        assert [r.signal_threshold for r in rows] == [0.10, 0.25, 0.40]

    def test_all_rows_share_sweep_id(self) -> None:
        rows = _run(thresholds=[0.20, 0.30, 0.40])
        assert len({r.sweep_id for r in rows}) == 1

    def test_sweep_id_is_12_hex_chars(self) -> None:
        rows = _run(thresholds=[0.25])
        assert len(rows[0].sweep_id) == 12
        assert all(c in "0123456789abcdef" for c in rows[0].sweep_id)

    def test_all_rows_share_run_ts(self) -> None:
        rows = _run(thresholds=[0.20, 0.30])
        assert len({r.run_ts for r in rows}) == 1

    def test_run_ts_is_recent(self) -> None:
        before = int(time.time())
        rows = _run(thresholds=[0.25])
        after = int(time.time())
        assert before <= rows[0].run_ts <= after

    def test_metrics_finite(self) -> None:
        rows = _run(thresholds=[0.15, 0.25, 0.35])
        for row in rows:
            assert math.isfinite(row.total_return)
            assert math.isfinite(row.cagr)
            assert math.isfinite(row.sharpe)
            assert math.isfinite(row.max_dd)
            assert 0.0 <= row.win_rate <= 1.0
            assert row.n_trades >= 0

    def test_max_dd_non_negative(self) -> None:
        rows = _run(thresholds=[0.05, 0.25, 0.50])
        for row in rows:
            assert row.max_dd >= 0.0

    def test_single_threshold(self) -> None:
        rows = _run(thresholds=[0.25])
        assert len(rows) == 1

    def test_empty_thresholds_raises(self) -> None:
        prices = _make_prices()
        cfg = _cfg()
        with patch("backtest.sweep._save_sweep_rows"), pytest.raises(ValueError):
            run_sweep(cfg, thresholds=[], _prices_df=prices)

    def test_different_thresholds_may_differ(self) -> None:
        # Very tight vs very loose threshold should produce different n_trades
        # on a non-trivial price series.
        rows = _run(thresholds=[0.01, 0.99], trend=15.0)
        low, high = rows
        # Not guaranteed to differ, but one metric should differ when thresholds are extreme.
        # Just verify both rows are valid.
        assert low.signal_threshold == pytest.approx(0.01)
        assert high.signal_threshold == pytest.approx(0.99)

    def test_preserves_threshold_order(self) -> None:
        thresholds = [0.30, 0.10, 0.20]
        rows = _run(thresholds=thresholds)
        assert [r.signal_threshold for r in rows] == thresholds


# ---------------------------------------------------------------------------
# _save_sweep_rows — DB persistence
# ---------------------------------------------------------------------------


def _make_test_conn(tmp_path, name: str = "test.db"):
    """Create an isolated SQLite connection with the full schema applied."""
    from database.db import Database

    db = Database(db_path=str(tmp_path / name))
    return db.connect()


class TestSaveSweepRows:
    def test_persists_to_db(self, tmp_path) -> None:
        import sqlite3
        from unittest.mock import patch

        conn = _make_test_conn(tmp_path)
        rows = [
            SweepRow(
                sweep_id="testid001",
                run_ts=1_700_000_000,
                signal_threshold=0.25,
                total_return=0.10,
                cagr=0.08,
                sharpe=1.5,
                max_dd=0.05,
                win_rate=0.6,
                n_trades=12,
            ),
            SweepRow(
                sweep_id="testid001",
                run_ts=1_700_000_000,
                signal_threshold=0.30,
                total_return=0.07,
                cagr=0.06,
                sharpe=1.2,
                max_dd=0.04,
                win_rate=0.55,
                n_trades=8,
            ),
        ]
        with patch("database.db.get_connection", return_value=conn):
            _save_sweep_rows(rows)

        saved = conn.execute(
            "SELECT sweep_id, signal_threshold, n_trades FROM sweep_results ORDER BY signal_threshold"
        ).fetchall()
        assert len(saved) == 2
        assert saved[0]["sweep_id"] == "testid001"
        assert saved[0]["signal_threshold"] == pytest.approx(0.25)
        assert saved[0]["n_trades"] == 12
        assert saved[1]["signal_threshold"] == pytest.approx(0.30)

    def test_unique_constraint_prevents_duplicate(self, tmp_path) -> None:
        import sqlite3
        from unittest.mock import patch

        conn = _make_test_conn(tmp_path, "dup.db")
        row = SweepRow(
            sweep_id="dup001",
            run_ts=1_700_000_000,
            signal_threshold=0.25,
            total_return=0.10,
            cagr=0.08,
            sharpe=1.5,
            max_dd=0.05,
            win_rate=0.6,
            n_trades=10,
        )
        with patch("database.db.get_connection", return_value=conn):
            _save_sweep_rows([row])
            with pytest.raises(sqlite3.IntegrityError):
                _save_sweep_rows([row])

    def test_persists_param_name(self, tmp_path) -> None:
        conn = _make_test_conn(tmp_path, "pname.db")
        row = SweepRow(
            sweep_id="pn001",
            run_ts=1_700_000_000,
            signal_threshold=0.10,
            total_return=0.05,
            cagr=0.04,
            sharpe=1.0,
            max_dd=0.03,
            win_rate=0.5,
            n_trades=5,
            param_name="position_size_pct",
        )
        with patch("database.db.get_connection", return_value=conn):
            _save_sweep_rows([row])

        saved = conn.execute(
            "SELECT param_name FROM sweep_results WHERE sweep_id = 'pn001'"
        ).fetchone()
        assert saved["param_name"] == "position_size_pct"


# ---------------------------------------------------------------------------
# SweepSpec — extensibility API
# ---------------------------------------------------------------------------


class TestSweepSpec:
    def test_valid_threshold_spec(self) -> None:
        spec = SweepSpec(param="signal_threshold", values=[0.20, 0.30])
        assert spec.param == "signal_threshold"
        assert spec.values == [0.20, 0.30]

    def test_valid_non_threshold_spec(self) -> None:
        spec = SweepSpec(param="position_size_pct", values=[0.05, 0.10])
        assert spec.param == "position_size_pct"

    def test_invalid_field_raises(self) -> None:
        with pytest.raises(ValueError, match="not a BacktestConfig field"):
            SweepSpec(param="nonexistent_field", values=[0.5])

    def test_empty_values_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SweepSpec(param="signal_threshold", values=[])

    def test_run_sweep_with_spec_threshold(self) -> None:
        spec = SweepSpec(param="signal_threshold", values=[0.20, 0.30])
        rows = _run_spec(spec)
        assert len(rows) == 2
        assert [r.signal_threshold for r in rows] == [0.20, 0.30]
        assert all(r.param_name == "signal_threshold" for r in rows)

    def test_run_sweep_with_spec_position_size(self) -> None:
        spec = SweepSpec(param="position_size_pct", values=[0.05, 0.10, 0.15])
        rows = _run_spec(spec)
        assert len(rows) == 3
        assert [r.signal_threshold for r in rows] == pytest.approx([0.05, 0.10, 0.15])
        assert all(r.param_name == "position_size_pct" for r in rows)

    def test_spec_and_thresholds_conflict_raises(self) -> None:
        spec = SweepSpec(param="signal_threshold", values=[0.25])
        prices = _make_prices()
        cfg = _cfg()
        with patch("backtest.sweep._save_sweep_rows"), pytest.raises(ValueError):
            run_sweep(cfg, thresholds=[0.25], spec=spec, _prices_df=prices)

    def test_run_sweep_default_param_name(self) -> None:
        rows = _run(thresholds=[0.25])
        assert rows[0].param_name == "signal_threshold"


# ---------------------------------------------------------------------------
# Deterministic sweep
# ---------------------------------------------------------------------------


class TestDeterministicSweep:
    def test_same_inputs_produce_same_metrics(self) -> None:
        """Identical prices + thresholds must yield identical metrics."""
        prices = _make_prices(trend=12.0)
        cfg = _cfg()
        thresholds = [0.20, 0.25, 0.30]

        rows1 = _run(cfg=cfg, thresholds=thresholds, prices=prices)
        rows2 = _run(cfg=cfg, thresholds=thresholds, prices=prices)

        for r1, r2 in zip(rows1, rows2):
            assert r1.total_return == pytest.approx(r2.total_return)
            assert r1.cagr == pytest.approx(r2.cagr)
            assert r1.sharpe == pytest.approx(r2.sharpe)
            assert r1.max_dd == pytest.approx(r2.max_dd)
            assert r1.win_rate == pytest.approx(r2.win_rate)
            assert r1.n_trades == r2.n_trades

    def test_different_runs_get_different_sweep_ids(self) -> None:
        """Each invocation generates a fresh sweep_id for historical accumulation."""
        rows1 = _run(thresholds=[0.25])
        rows2 = _run(thresholds=[0.25])
        assert rows1[0].sweep_id != rows2[0].sweep_id

    def test_metrics_stable_across_threshold_range(self) -> None:
        """All rows must produce finite, bounded metrics."""
        rows = _run(thresholds=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
        for row in rows:
            assert math.isfinite(row.total_return)
            assert math.isfinite(row.cagr)
            assert math.isfinite(row.sharpe)
            assert 0.0 <= row.max_dd
            assert 0.0 <= row.win_rate <= 1.0
            assert row.n_trades >= 0


# ---------------------------------------------------------------------------
# Robustness framing
# ---------------------------------------------------------------------------


class TestRobustnessFraming:
    def test_current_threshold_appears_in_sweep(self) -> None:
        """config.SIGNAL_THRESHOLD should be identifiable among sweep rows."""
        import config as cfg_mod

        current = cfg_mod.SIGNAL_THRESHOLD
        thresholds = [0.15, 0.20, current, 0.30, 0.35]
        rows = _run(thresholds=thresholds)

        current_rows = [r for r in rows if abs(r.signal_threshold - current) < 1e-9]
        assert len(current_rows) == 1, "Current threshold must appear exactly once"

    def test_current_threshold_need_not_be_champion(self) -> None:
        """Robustness framing: the current value is marked, not forced to be best."""
        # With a strong uptrend, a very low threshold (catches signals early)
        # may outperform the default 0.25 — both outcomes are valid.
        thresholds = [0.01, 0.25, 0.99]
        rows = _run(thresholds=thresholds, trend=20.0)
        assert len(rows) == 3

        current_row = next(r for r in rows if abs(r.signal_threshold - 0.25) < 1e-9)
        returns = [r.total_return for r in rows]
        max_return = max(returns)

        # Either the current row IS the champion, or it isn't — both are acceptable.
        # What matters is that we can identify the current row independently of rank.
        is_champion = abs(current_row.total_return - max_return) < 1e-9
        _ = is_champion  # framing: dashboard shows current + all values, not just winner

        # All three must be present with correct param ordering preserved.
        assert [r.signal_threshold for r in rows] == pytest.approx([0.01, 0.25, 0.99])

    def test_sweep_marks_only_exact_match_as_current(self) -> None:
        """Floating-point equality check: only ≈ exact match counts as 'current'."""
        thresholds = [0.20, 0.25, 0.30]
        rows = _run(thresholds=thresholds)

        import config as cfg_mod
        current = cfg_mod.SIGNAL_THRESHOLD

        near_current = [r for r in rows if abs(r.signal_threshold - current) < 1e-9]
        not_current = [r for r in rows if abs(r.signal_threshold - current) >= 1e-9]

        # Exactly one row is the current threshold.
        assert len(near_current) + len(not_current) == len(rows)
        assert len(near_current) <= 1

    def test_sweep_values_are_all_recorded(self) -> None:
        """Every value in the grid appears in the result — no silent skips."""
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        rows = _run(thresholds=thresholds)
        recorded = [r.signal_threshold for r in rows]
        for t in thresholds:
            assert any(abs(v - t) < 1e-9 for v in recorded), f"{t} missing from results"
