"""Tests for engine/stocks_executor.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import json

import pytest

import config
from engine.stocks_executor import (
    StocksExecutor,
    StocksOrderResult,
    _get_open_position,
    _latest_close,
    execute_stocks_once,
)
from engine.risk_manager import RiskManager
from engine.signal_aggregator import AggregatedSignal


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


@pytest.fixture(autouse=True)
def _patch_db(temp_db: str, monkeypatch):
    import database.db as db_module

    class _FakeDB:
        def connect(self):
            c = sqlite3.connect(temp_db)
            c.row_factory = sqlite3.Row
            return c

    monkeypatch.setattr(db_module, "_db", _FakeDB())
    yield


@pytest.fixture()
def executor() -> StocksExecutor:
    return StocksExecutor()


def _make_signal(
    symbol: str = "AAPL",
    asset_class: str = "stocks",
    action: str = "buy",
    signal_id: int = 1,
) -> AggregatedSignal:
    return AggregatedSignal(
        symbol=symbol,
        asset_class=asset_class,
        signal_id=signal_id,
        momentum_score=0.5,
        volatility_score=0.3,
        funding_score=None,
        sentiment_score=0.2,
        composite_score=0.35,
        action=action,
    )


def _insert_price(db_path: str, symbol: str, close: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO prices
            (symbol, asset_class, timeframe, ts, open, high, low, close, volume)
        VALUES (?, 'stocks', '1h', ?, ?, ?, ?, ?, 500000.0)
        """,
        (symbol, int(time.time()), close, close, close, close),
    )
    conn.commit()
    conn.close()


def _open_position_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE status = 'open'"
    ).fetchone()
    conn.close()
    return row[0]


def _order_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
    conn.close()
    return row[0]


# ---------------------------------------------------------------------------
# _latest_close
# ---------------------------------------------------------------------------

class TestLatestClose:
    def test_returns_none_when_no_rows(self):
        assert _latest_close("AAPL") is None

    def test_returns_close_price(self, temp_db):
        _insert_price(temp_db, "AAPL", 150.0)
        assert _latest_close("AAPL") == pytest.approx(150.0)

    def test_returns_most_recent(self, temp_db):
        conn = sqlite3.connect(temp_db)
        for ts, price in [(1000, 140.0), (2000, 160.0)]:
            conn.execute(
                """INSERT INTO prices
                       (symbol, asset_class, timeframe, ts, open, high, low, close, volume)
                   VALUES ('AAPL', 'stocks', '1h', ?, ?, ?, ?, ?, 1.0)""",
                (ts, price, price, price, price),
            )
        conn.commit()
        conn.close()
        assert _latest_close("AAPL") == pytest.approx(160.0)


# ---------------------------------------------------------------------------
# Paper buy
# ---------------------------------------------------------------------------

class TestPaperBuy:
    def test_buy_creates_position_and_order(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        sig = _make_signal(action="buy")
        result = executor.execute(sig, equity=10_000.0)

        assert result is not None
        assert result.side == "buy"
        assert result.symbol == "AAPL"
        assert result.asset_class == "stocks"
        assert result.is_paper is True
        assert result.exchange_id is None
        assert result.price == pytest.approx(150.0)
        assert result.quantity == pytest.approx(10_000.0 * config.POSITION_SIZE_PCT / 150.0)

        assert _open_position_count(temp_db) == 1
        assert _order_count(temp_db) == 1

    def test_buy_order_is_paper_flagged(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        executor.execute(_make_signal(action="buy"), equity=5_000.0)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT is_paper FROM orders").fetchone()
        conn.close()
        assert row[0] == 1

    def test_position_is_paper_flagged(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        executor.execute(_make_signal(action="buy"), equity=5_000.0)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT is_paper FROM positions").fetchone()
        conn.close()
        assert row[0] == 1

    def test_position_side_is_long(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        executor.execute(_make_signal(action="buy"), equity=5_000.0)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT side FROM positions").fetchone()
        conn.close()
        assert row[0] == "long"


# ---------------------------------------------------------------------------
# Paper sell
# ---------------------------------------------------------------------------

class TestPaperSell:
    def _setup_open_position(self, db_path: str, symbol: str, entry_price: float) -> int:
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            """
            INSERT INTO positions
                (symbol, asset_class, side, quantity, entry_price, current_price, is_paper)
            VALUES (?, 'stocks', 'long', 10.0, ?, ?, 1)
            """,
            (symbol, entry_price, entry_price),
        )
        pos_id = cur.lastrowid
        conn.commit()
        conn.close()
        return pos_id

    def test_sell_closes_position(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        self._setup_open_position(temp_db, "AAPL", 140.0)
        _insert_price(temp_db, "AAPL", 155.0)
        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.side == "sell"
        assert result.is_paper is True
        assert result.exchange_id is None

        conn = sqlite3.connect(temp_db)
        pos = conn.execute("SELECT status, realized_pnl FROM positions").fetchone()
        conn.close()
        assert pos[0] == "closed"
        assert pos[1] == pytest.approx(10.0 * (155.0 - 140.0))

    def test_sell_with_loss(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        self._setup_open_position(temp_db, "AAPL", 160.0)
        _insert_price(temp_db, "AAPL", 145.0)
        executor.execute(_make_signal(action="sell"), equity=10_000.0)

        conn = sqlite3.connect(temp_db)
        pos = conn.execute("SELECT realized_pnl FROM positions").fetchone()
        conn.close()
        assert pos[0] == pytest.approx(10.0 * (145.0 - 160.0))  # negative

    def test_sell_no_open_position_returns_none(self, executor, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)
        assert result is None

    def test_sell_order_created_in_db(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        self._setup_open_position(temp_db, "AAPL", 140.0)
        _insert_price(temp_db, "AAPL", 150.0)
        executor.execute(_make_signal(action="sell"), equity=10_000.0)

        conn = sqlite3.connect(temp_db)
        order = conn.execute("SELECT side, status FROM orders").fetchone()
        conn.close()
        assert order[0] == "sell"
        assert order[1] == "filled"


# ---------------------------------------------------------------------------
# Hold / non-stocks
# ---------------------------------------------------------------------------

class TestHoldAndNonStocks:
    def test_hold_returns_none(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        result = executor.execute(_make_signal(action="hold"), equity=10_000.0)
        assert result is None

    def test_hold_creates_no_db_rows(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        executor.execute(_make_signal(action="hold"), equity=10_000.0)
        assert _open_position_count(temp_db) == 0
        assert _order_count(temp_db) == 0

    def test_crypto_signal_skipped(self, executor, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        sig = _make_signal(asset_class="crypto", action="buy")
        result = executor.execute(sig, equity=10_000.0)
        assert result is None


# ---------------------------------------------------------------------------
# Risk blocks
# ---------------------------------------------------------------------------

class TestRiskBlocks:
    def test_drawdown_block_prevents_buy(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        rm = executor._rm
        rm.record_equity(10_000.0)
        rm.record_equity(8_000.0)  # 20 % drawdown > 15 % limit

        result = executor.execute(_make_signal(action="buy"), equity=8_000.0)
        assert result is None
        assert _open_position_count(temp_db) == 0

    def test_position_cap_block_prevents_buy(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        conn = sqlite3.connect(temp_db)
        for i in range(config.MAX_OPEN_POSITIONS):
            conn.execute(
                """INSERT INTO positions
                       (symbol, asset_class, side, quantity, entry_price,
                        current_price, status, is_paper)
                   VALUES (?, 'stocks', 'long', 10.0, 100.0, 100.0, 'open', 1)""",
                (f"SYM{i}",),
            )
        conn.commit()
        conn.close()

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None

    def test_duplicate_open_position_skipped(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('AAPL', 'stocks', 'long', 10.0, 150.0, 150.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None
        assert _open_position_count(temp_db) == 1  # unchanged

    def test_no_price_prevents_buy(self, executor, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None

    def test_no_price_prevents_sell(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('AAPL', 'stocks', 'long', 10.0, 150.0, 150.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)
        assert result is None


# ---------------------------------------------------------------------------
# Live mode (STOCKS_LIVE=true)
# ---------------------------------------------------------------------------

class TestLiveMode:
    def _mock_alpaca(self, monkeypatch, order_id: str = "AP-001", avg_price: float = 151.0):
        import urllib.request as urllib_request

        class _FakeResponse:
            def __init__(self):
                self._data = json.dumps({
                    "id": order_id,
                    "filled_avg_price": str(avg_price),
                }).encode()

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        monkeypatch.setattr(
            "engine.stocks_executor.urlopen",
            lambda req, timeout=None: _FakeResponse(),
        )

    def test_live_buy_calls_alpaca(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        _insert_price(temp_db, "AAPL", 150.0)
        self._mock_alpaca(monkeypatch, order_id="AP-001", avg_price=151.0)

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is False
        assert result.exchange_id == "AP-001"
        assert result.price == pytest.approx(151.0)

    def test_live_sell_calls_alpaca(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('AAPL', 'stocks', 'long', 10.0, 140.0, 140.0, 'open', 0)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "AAPL", 155.0)
        self._mock_alpaca(monkeypatch, order_id="AP-002", avg_price=155.5)

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is False
        assert result.exchange_id == "AP-002"
        assert result.price == pytest.approx(155.5)

    def test_live_order_failure_returns_none(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        _insert_price(temp_db, "AAPL", 150.0)
        monkeypatch.setattr(
            "engine.stocks_executor.urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(OSError("network error")),
        )

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None
        assert _open_position_count(temp_db) == 0

    def test_live_buy_is_not_paper_in_db(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        _insert_price(temp_db, "AAPL", 150.0)
        self._mock_alpaca(monkeypatch)
        executor.execute(_make_signal(action="buy"), equity=10_000.0)

        conn = sqlite3.connect(temp_db)
        order = conn.execute("SELECT is_paper FROM orders").fetchone()
        pos = conn.execute("SELECT is_paper FROM positions").fetchone()
        conn.close()
        assert order[0] == 0
        assert pos[0] == 0

    def test_paper_position_closed_as_paper_when_live_toggled(
        self, executor, temp_db, monkeypatch
    ):
        """Flipping STOCKS_LIVE=true must not send a live sell for a paper position."""
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('AAPL', 'stocks', 'long', 10.0, 140.0, 140.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "AAPL", 155.0)
        alpaca_called = []
        monkeypatch.setattr(
            "engine.stocks_executor.urlopen",
            lambda req, timeout=None: alpaca_called.append(req) or (_ for _ in ()).throw(AssertionError("must not call Alpaca")),
        )

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is True
        assert result.exchange_id is None
        assert alpaca_called == []

    def test_live_position_closed_as_live_when_toggle_off(
        self, executor, temp_db, monkeypatch
    ):
        """STOCKS_LIVE=false must not skip the live sell for a position opened live."""
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('AAPL', 'stocks', 'long', 10.0, 140.0, 140.0, 'open', 0)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "AAPL", 155.0)
        self._mock_alpaca(monkeypatch, order_id="AP-999", avg_price=155.5)

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is False
        assert result.exchange_id == "AP-999"


# ---------------------------------------------------------------------------
# execute_stocks_once
# ---------------------------------------------------------------------------

class TestExecuteStocksOnce:
    def test_empty_signals(self, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        results = execute_stocks_once([], equity=10_000.0)
        assert results == []

    def test_holds_excluded(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        sigs = [_make_signal(action="hold"), _make_signal(symbol="MSFT", action="hold")]
        results = execute_stocks_once(sigs, equity=10_000.0)
        assert results == []

    def test_multiple_buys(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        _insert_price(temp_db, "MSFT", 300.0)
        sigs = [
            _make_signal("AAPL", action="buy", signal_id=1),
            _make_signal("MSFT", action="buy", signal_id=2),
        ]
        results = execute_stocks_once(sigs, equity=100_000.0)
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert symbols == {"AAPL", "MSFT"}

    def test_accepts_custom_executor(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        _insert_price(temp_db, "AAPL", 150.0)
        custom_ex = StocksExecutor()
        results = execute_stocks_once(
            [_make_signal(action="buy")], equity=10_000.0, executor=custom_ex
        )
        assert len(results) == 1
