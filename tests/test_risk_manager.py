"""Tests for engine/risk_manager.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import config
from engine.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path: Path) -> str:
    db_file = str(tmp_path / "test.db")
    schema = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()
    conn = sqlite3.connect(db_file)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture()
def rm(temp_db: str, monkeypatch):
    """RiskManager wired to a fresh in-memory-like test DB."""
    import database.db as db_module

    class _FakeDB:
        def connect(self):
            c = sqlite3.connect(temp_db)
            c.row_factory = sqlite3.Row
            return c

    monkeypatch.setattr(db_module, "_db", _FakeDB())
    return RiskManager()


# ---------------------------------------------------------------------------
# position_size
# ---------------------------------------------------------------------------

class TestPositionSize:
    def test_ten_percent_of_equity(self, rm: RiskManager):
        # config.POSITION_SIZE_PCT defaults to 0.10
        assert rm.position_size(10_000) == pytest.approx(10_000 * config.POSITION_SIZE_PCT)

    def test_zero_equity(self, rm: RiskManager):
        assert rm.position_size(0.0) == 0.0


# ---------------------------------------------------------------------------
# record_equity + _current_drawdown
# ---------------------------------------------------------------------------

class TestRecordEquity:
    def test_first_snapshot_has_zero_drawdown(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        assert rm._current_drawdown() == pytest.approx(0.0)

    def test_peak_rises_with_equity(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(12_000.0)
        assert rm._current_drawdown() == pytest.approx(0.0)

    def test_drawdown_computed_correctly(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(8_500.0)
        # drawdown = (10_000 - 8_500) / 10_000 = 0.15
        assert rm._current_drawdown() == pytest.approx(0.15)

    def test_peak_does_not_decrease(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(8_000.0)
        # peak still 10_000; equity rises but not past 10_000
        rm.record_equity(9_000.0)
        assert rm._current_drawdown() == pytest.approx(0.10)

    def test_is_paper_flag_stored(self, rm: RiskManager, temp_db: str):
        rm.record_equity(5_000.0, is_paper=False)
        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT is_paper FROM equity_curve ORDER BY ts DESC LIMIT 1").fetchone()
        conn.close()
        assert row[0] == 0


# ---------------------------------------------------------------------------
# can_open — happy path
# ---------------------------------------------------------------------------

class TestCanOpenAllowed:
    def test_no_history_is_allowed(self, rm: RiskManager):
        allowed, reason = rm.can_open("BTC/USDT", "crypto")
        assert allowed is True
        assert reason == ""

    def test_small_drawdown_is_allowed(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(9_000.0)  # 10 % drawdown — under limit
        allowed, reason = rm.can_open("ETH/USDT", "crypto")
        assert allowed is True

    def test_two_open_positions_allowed(self, rm: RiskManager, temp_db: str):
        conn = sqlite3.connect(temp_db)
        for sym in ("BTC/USDT", "ETH/USDT"):
            conn.execute(
                """INSERT INTO positions (symbol, asset_class, side, quantity,
                   entry_price, current_price, status)
                   VALUES (?, 'crypto', 'long', 1.0, 100.0, 100.0, 'open')""",
                (sym,),
            )
        conn.commit()
        conn.close()
        allowed, _ = rm.can_open("SOL/USDT", "crypto")
        assert allowed is True


# ---------------------------------------------------------------------------
# can_open — blocked by drawdown
# ---------------------------------------------------------------------------

class TestCanOpenDrawdownBlock:
    def test_exactly_at_limit_is_blocked(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(8_500.0)  # exactly 15 % drawdown
        allowed, reason = rm.can_open("BTC/USDT", "crypto")
        assert allowed is False
        assert "drawdown" in reason

    def test_above_limit_is_blocked(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(8_000.0)  # 20 % drawdown
        allowed, reason = rm.can_open("AAPL", "stocks")
        assert allowed is False
        assert "drawdown" in reason

    def test_reason_contains_percentages(self, rm: RiskManager):
        rm.record_equity(10_000.0)
        rm.record_equity(8_000.0)
        _, reason = rm.can_open("BTC/USDT", "crypto")
        assert "%" in reason


# ---------------------------------------------------------------------------
# can_open — blocked by position cap
# ---------------------------------------------------------------------------

class TestCanOpenPositionCapBlock:
    def _insert_open_positions(self, db_path: str, n: int) -> None:
        conn = sqlite3.connect(db_path)
        for i in range(n):
            conn.execute(
                """INSERT INTO positions (symbol, asset_class, side, quantity,
                   entry_price, current_price, status)
                   VALUES (?, 'crypto', 'long', 1.0, 100.0, 100.0, 'open')""",
                (f"SYM{i}/USDT",),
            )
        conn.commit()
        conn.close()

    def test_at_max_positions_is_blocked(self, rm: RiskManager, temp_db: str):
        self._insert_open_positions(temp_db, config.MAX_OPEN_POSITIONS)
        allowed, reason = rm.can_open("NEW/USDT", "crypto")
        assert allowed is False
        assert "open positions" in reason

    def test_reason_contains_count(self, rm: RiskManager, temp_db: str):
        self._insert_open_positions(temp_db, config.MAX_OPEN_POSITIONS)
        _, reason = rm.can_open("NEW/USDT", "crypto")
        assert str(config.MAX_OPEN_POSITIONS) in reason

    def test_closed_positions_not_counted(self, rm: RiskManager, temp_db: str):
        # Fill up with closed positions — should not block
        conn = sqlite3.connect(temp_db)
        for i in range(10):
            conn.execute(
                """INSERT INTO positions (symbol, asset_class, side, quantity,
                   entry_price, current_price, status)
                   VALUES (?, 'crypto', 'long', 1.0, 100.0, 100.0, 'closed')""",
                (f"SYM{i}/USDT",),
            )
        conn.commit()
        conn.close()
        allowed, _ = rm.can_open("BTC/USDT", "crypto")
        assert allowed is True

    def test_drawdown_checked_before_positions(self, rm: RiskManager, temp_db: str):
        # Both limits triggered — drawdown message should appear (checked first)
        rm.record_equity(10_000.0)
        rm.record_equity(8_000.0)  # 20 % drawdown
        self._insert_open_positions(temp_db, config.MAX_OPEN_POSITIONS)
        allowed, reason = rm.can_open("BTC/USDT", "crypto")
        assert allowed is False
        assert "drawdown" in reason
