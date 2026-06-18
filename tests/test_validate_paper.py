"""Tests for validate_paper.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import config


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SCHEMA = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    p = str(tmp_path / "test.db")
    conn = sqlite3.connect(p)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    return p


@pytest.fixture(autouse=True)
def _reset_db_singleton(monkeypatch, db_path: str):
    """Wire the module-level DB singleton to the test database."""
    import database.db as db_module

    class _TestDB:
        def connect(self):
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            return c

    monkeypatch.setattr(db_module, "_db", _TestDB())


def _insert_equity(db: str, ts: int, equity: float, peak: float, dd: float, is_paper: int = 1):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO equity_curve (ts, equity, peak_equity, drawdown_pct, is_paper) VALUES (?,?,?,?,?)",
        (ts, equity, peak, dd, is_paper),
    )
    conn.commit()
    conn.close()


def _insert_signal(db: str, symbol: str = "BTC/USDT", ts: int | None = None):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO signals (symbol, asset_class, ts, momentum_score, volatility_score, composite_score, action) "
        "VALUES (?, 'crypto', ?, 0.1, 0.1, 0.1, 'hold')",
        (symbol, ts or int(time.time())),
    )
    conn.commit()
    conn.close()


def _insert_order(db: str, is_paper: int = 1):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO orders (symbol, asset_class, side, quantity, price, is_paper) "
        "VALUES ('BTC/USDT', 'crypto', 'buy', 0.01, 50000.0, ?)",
        (is_paper,),
    )
    conn.commit()
    conn.close()


def _span_equity(db: str, hours: float, equity: float = 10_000.0):
    """Insert two equity snapshots spanning *hours*."""
    now = int(time.time())
    start = now - int(hours * 3600)
    _insert_equity(db, start, equity, equity, 0.0)
    _insert_equity(db, now, equity, equity, 0.0)


# ---------------------------------------------------------------------------
# Duration checks
# ---------------------------------------------------------------------------

class TestDuration:
    def test_no_snapshots_fails(self, db_path):
        from validate_paper import validate
        r = validate(db_path=db_path)
        assert r.passed is False
        assert any("no paper equity snapshots" in f for f in r.failures)

    def test_under_48h_fails(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=24.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert r.passed is False
        assert any("24" in f or "duration" in f for f in r.failures)

    def test_exactly_48h_passes_duration(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        # Duration check passes (failures should not mention duration)
        assert not any("duration" in f for f in r.failures)

    def test_over_48h_passes_duration(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=72.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert not any("duration" in f for f in r.failures)

    def test_duration_hours_reported_correctly(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=50.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert r.duration_hours == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# Signal checks
# ---------------------------------------------------------------------------

class TestSignals:
    def test_no_signals_fails(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        r = validate(db_path=db_path)
        assert r.passed is False
        assert any("signals" in f for f in r.failures)

    def test_signals_present_passes_signal_check(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert not any("no deterministic signals" in f for f in r.failures)

    def test_signal_count_reported(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        for _ in range(5):
            _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert r.signals_generated == 5


# ---------------------------------------------------------------------------
# Live order leak detection
# ---------------------------------------------------------------------------

class TestLiveOrderLeak:
    def test_paper_orders_no_leak(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        _insert_signal(db_path)
        _insert_order(db_path, is_paper=1)
        r = validate(db_path=db_path)
        assert r.live_order_leak is False

    def test_live_order_detected(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        _insert_signal(db_path)
        _insert_order(db_path, is_paper=0)
        r = validate(db_path=db_path)
        assert r.live_order_leak is True
        assert r.passed is False
        assert any("live order" in f for f in r.failures)

    def test_orders_placed_count(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        _insert_signal(db_path)
        for _ in range(3):
            _insert_order(db_path, is_paper=1)
        r = validate(db_path=db_path)
        assert r.orders_placed == 3


# ---------------------------------------------------------------------------
# Drawdown reporting
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_zero_drawdown_no_failure(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=48.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert r.max_drawdown_pct == pytest.approx(0.0)
        assert not any("drawdown" in f for f in r.failures)

    def test_high_drawdown_noted_in_failures(self, db_path):
        from validate_paper import validate
        now = int(time.time())
        _insert_equity(db_path, now - 200_000, 10_000.0, 10_000.0, 0.0)
        _insert_equity(db_path, now, 8_500.0, 10_000.0, 0.15)  # exactly at limit
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert r.max_drawdown_pct == pytest.approx(0.15)
        assert any("drawdown" in f for f in r.failures)

    def test_drawdown_below_limit_no_failure(self, db_path):
        from validate_paper import validate
        now = int(time.time())
        _insert_equity(db_path, now - 200_000, 10_000.0, 10_000.0, 0.0)
        _insert_equity(db_path, now, 9_000.0, 10_000.0, 0.10)  # 10 % — under limit
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert not any("drawdown" in f for f in r.failures)


# ---------------------------------------------------------------------------
# Full passing scenario
# ---------------------------------------------------------------------------

class TestFullPass:
    def test_all_conditions_met(self, db_path):
        from validate_paper import validate
        _span_equity(db_path, hours=50.0)
        _insert_signal(db_path)
        _insert_order(db_path, is_paper=1)
        r = validate(db_path=db_path)
        assert r.passed is True
        assert r.failures == []
        assert r.live_order_leak is False

    def test_cycles_run_count(self, db_path):
        from validate_paper import validate
        now = int(time.time())
        for i in range(5):
            ts = now - (50 * 3600) + i * 3600
            _insert_equity(db_path, ts, 10_000.0, 10_000.0, 0.0)
        _insert_signal(db_path)
        r = validate(db_path=db_path)
        assert r.cycles_run == 5


# ---------------------------------------------------------------------------
# print_report — smoke test (no crash, correct exit indication)
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_passed_report_prints(self, capsys):
        from validate_paper import PaperValidationReport, print_report
        r = PaperValidationReport(
            passed=True,
            duration_hours=50.0,
            cycles_run=50,
            signals_generated=150,
            orders_placed=10,
            realized_pnl=200.0,
            max_drawdown_pct=0.05,
            live_order_leak=False,
        )
        print_report(r)
        out = capsys.readouterr().out
        assert "PASSED" in out
        assert "50.0 h" in out

    def test_failed_report_prints_failures(self, capsys):
        from validate_paper import PaperValidationReport, print_report
        r = PaperValidationReport(
            passed=False,
            duration_hours=10.0,
            cycles_run=10,
            signals_generated=0,
            orders_placed=0,
            realized_pnl=0.0,
            max_drawdown_pct=0.0,
            live_order_leak=False,
            failures=["paper run duration 10.0 h < required 48 h"],
        )
        print_report(r)
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "duration" in out
