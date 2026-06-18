"""Tests for engine/order_executor.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import config
from engine.order_executor import (
    OrderExecutor,
    OrderResult,
    _get_open_position,
    _latest_close,
    execute_once,
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
def executor() -> OrderExecutor:
    return OrderExecutor()


def _make_signal(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    action: str = "buy",
    signal_id: int = 1,
) -> AggregatedSignal:
    return AggregatedSignal(
        symbol=symbol,
        asset_class=asset_class,
        signal_id=signal_id,
        momentum_score=0.5,
        volatility_score=0.3,
        funding_score=0.1,
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
        VALUES (?, 'crypto', '1h', ?, ?, ?, ?, ?, 1000.0)
        """,
        (symbol, int(time.time()), close, close, close, close),
    )
    conn.commit()
    conn.close()


def _insert_signal_row(db_path: str, symbol: str) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """
        INSERT INTO signals
            (symbol, asset_class, ts, momentum_score, volatility_score,
             composite_score, action)
        VALUES (?, 'crypto', ?, 0.5, 0.3, 0.35, 'buy')
        """,
        (symbol, int(time.time())),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


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
        assert _latest_close("BTC/USDT") is None

    def test_returns_close_price(self, temp_db):
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        assert _latest_close("BTC/USDT") == pytest.approx(50_000.0)

    def test_returns_most_recent(self, temp_db):
        conn = sqlite3.connect(temp_db)
        for ts, price in [(1000, 40_000.0), (2000, 55_000.0)]:
            conn.execute(
                """INSERT INTO prices
                       (symbol, asset_class, timeframe, ts, open, high, low, close, volume)
                   VALUES ('BTC/USDT', 'crypto', '1h', ?, ?, ?, ?, ?, 1.0)""",
                (ts, price, price, price, price),
            )
        conn.commit()
        conn.close()
        assert _latest_close("BTC/USDT") == pytest.approx(55_000.0)


# ---------------------------------------------------------------------------
# Paper buy
# ---------------------------------------------------------------------------

class TestPaperBuy:
    def test_buy_creates_position_and_order(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        sig = _make_signal(action="buy")
        result = executor.execute(sig, equity=10_000.0)

        assert result is not None
        assert result.side == "buy"
        assert result.symbol == "BTC/USDT"
        assert result.is_paper is True
        assert result.exchange_id is None
        # paper price is inflated by slippage + fee
        expected_price = 50_000.0 * (1.0 + config.SLIPPAGE_PCT) * (1.0 + config.FEE_PCT)
        assert result.price == pytest.approx(expected_price)
        # quantity is sized on the raw market price, not the cost-adjusted one
        assert result.quantity == pytest.approx(10_000.0 * config.POSITION_SIZE_PCT / 50_000.0)

        assert _open_position_count(temp_db) == 1
        assert _order_count(temp_db) == 1

    def test_buy_order_is_paper_flagged(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 30_000.0)
        executor.execute(_make_signal(action="buy"), equity=5_000.0)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT is_paper FROM orders").fetchone()
        conn.close()
        assert row[0] == 1

    def test_position_is_paper_flagged(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 30_000.0)
        executor.execute(_make_signal(action="buy"), equity=5_000.0)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT is_paper FROM positions").fetchone()
        conn.close()
        assert row[0] == 1

    def test_position_side_is_long(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 30_000.0)
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
            VALUES (?, 'crypto', 'long', 0.2, ?, ?, 1)
            """,
            (symbol, entry_price, entry_price),
        )
        pos_id = cur.lastrowid
        conn.commit()
        conn.close()
        return pos_id

    def test_sell_closes_position(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        self._setup_open_position(temp_db, "BTC/USDT", 40_000.0)
        _insert_price(temp_db, "BTC/USDT", 45_000.0)
        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.side == "sell"
        assert result.is_paper is True
        assert result.exchange_id is None

        exec_close = 45_000.0 * (1.0 - config.SLIPPAGE_PCT) * (1.0 - config.FEE_PCT)
        conn = sqlite3.connect(temp_db)
        pos = conn.execute("SELECT status, realized_pnl FROM positions").fetchone()
        conn.close()
        assert pos[0] == "closed"
        assert pos[1] == pytest.approx(0.2 * (exec_close - 40_000.0))

    def test_sell_with_loss(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        self._setup_open_position(temp_db, "BTC/USDT", 50_000.0)
        _insert_price(temp_db, "BTC/USDT", 45_000.0)
        executor.execute(_make_signal(action="sell"), equity=10_000.0)

        exec_close = 45_000.0 * (1.0 - config.SLIPPAGE_PCT) * (1.0 - config.FEE_PCT)
        conn = sqlite3.connect(temp_db)
        pos = conn.execute("SELECT realized_pnl FROM positions").fetchone()
        conn.close()
        assert pos[0] == pytest.approx(0.2 * (exec_close - 50_000.0))  # negative

    def test_sell_no_open_position_returns_none(self, executor, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)
        assert result is None

    def test_sell_order_created_in_db(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        self._setup_open_position(temp_db, "BTC/USDT", 40_000.0)
        _insert_price(temp_db, "BTC/USDT", 42_000.0)
        executor.execute(_make_signal(action="sell"), equity=10_000.0)

        conn = sqlite3.connect(temp_db)
        order = conn.execute("SELECT side, status FROM orders").fetchone()
        conn.close()
        assert order[0] == "sell"
        assert order[1] == "filled"


# ---------------------------------------------------------------------------
# Hold / non-crypto
# ---------------------------------------------------------------------------

class TestHoldAndNonCrypto:
    def test_hold_returns_none(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        result = executor.execute(_make_signal(action="hold"), equity=10_000.0)
        assert result is None

    def test_hold_creates_no_db_rows(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        executor.execute(_make_signal(action="hold"), equity=10_000.0)
        assert _open_position_count(temp_db) == 0
        assert _order_count(temp_db) == 0

    def test_stocks_signal_skipped(self, executor, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        sig = _make_signal(asset_class="stocks", action="buy")
        result = executor.execute(sig, equity=10_000.0)
        assert result is None


# ---------------------------------------------------------------------------
# Risk blocks
# ---------------------------------------------------------------------------

class TestRiskBlocks:
    def test_drawdown_block_prevents_buy(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        rm = executor._rm
        rm.record_equity(10_000.0)
        rm.record_equity(8_000.0)  # 20 % drawdown > 15 % limit

        result = executor.execute(_make_signal(action="buy"), equity=8_000.0)
        assert result is None
        assert _open_position_count(temp_db) == 0

    def test_position_cap_block_prevents_buy(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        conn = sqlite3.connect(temp_db)
        for i in range(config.MAX_OPEN_POSITIONS):
            conn.execute(
                """INSERT INTO positions
                       (symbol, asset_class, side, quantity, entry_price,
                        current_price, status, is_paper)
                   VALUES (?, 'crypto', 'long', 1.0, 100.0, 100.0, 'open', 1)""",
                (f"SYM{i}/USDT",),
            )
        conn.commit()
        conn.close()

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None

    def test_duplicate_open_position_skipped(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.1, 50000.0, 50000.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None
        assert _open_position_count(temp_db) == 1  # unchanged

    def test_no_price_prevents_buy(self, executor, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None

    def test_no_price_prevents_sell(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.1, 50000.0, 50000.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)
        assert result is None


# ---------------------------------------------------------------------------
# Live mode (CRYPTO_LIVE=true)
# ---------------------------------------------------------------------------

class TestLiveMode:
    def _mock_exchange(self, monkeypatch, order_id: str = "123", avg_price: float = 50_500.0):
        mock_ex = MagicMock()
        mock_ex.amount_to_precision.side_effect = lambda sym, qty: qty
        mock_ex.create_market_order.return_value = {
            "id": order_id,
            "average": avg_price,
            "price": avg_price,
        }
        monkeypatch.setattr(
            "engine.order_executor._make_exchange", lambda: mock_ex
        )
        return mock_ex

    def test_live_buy_calls_binance(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        mock_ex = self._mock_exchange(monkeypatch, order_id="BN-001", avg_price=50_100.0)

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is False
        assert result.exchange_id == "BN-001"
        assert result.price == pytest.approx(50_100.0)
        mock_ex.create_market_order.assert_called_once()
        call_args = mock_ex.create_market_order.call_args
        assert call_args[0][0] == "BTC/USDT"
        assert call_args[0][1] == "buy"

    def test_live_sell_calls_binance(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.2, 48000.0, 48000.0, 'open', 0)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "BTC/USDT", 51_000.0)
        mock_ex = self._mock_exchange(monkeypatch, order_id="BN-002", avg_price=51_200.0)

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is False
        assert result.exchange_id == "BN-002"
        mock_ex.create_market_order.assert_called_once()
        assert mock_ex.create_market_order.call_args[0][1] == "sell"

    def test_live_order_failure_returns_none(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        mock_ex = MagicMock()
        mock_ex.amount_to_precision.side_effect = lambda sym, qty: qty
        mock_ex.create_market_order.side_effect = RuntimeError("network error")
        monkeypatch.setattr("engine.order_executor._make_exchange", lambda: mock_ex)

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)
        assert result is None
        assert _open_position_count(temp_db) == 0

    def test_live_buy_is_not_paper_in_db(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        self._mock_exchange(monkeypatch)
        executor.execute(_make_signal(action="buy"), equity=10_000.0)

        conn = sqlite3.connect(temp_db)
        order = conn.execute("SELECT is_paper FROM orders").fetchone()
        pos = conn.execute("SELECT is_paper FROM positions").fetchone()
        conn.close()
        assert order[0] == 0
        assert pos[0] == 0

    def test_lot_size_precision_applied(self, executor, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        mock_ex = MagicMock()
        mock_ex.amount_to_precision.return_value = 0.002  # rounded by exchange
        mock_ex.create_market_order.return_value = {
            "id": "99", "average": 50_000.0, "price": 50_000.0
        }
        monkeypatch.setattr("engine.order_executor._make_exchange", lambda: mock_ex)

        executor.execute(_make_signal(action="buy"), equity=10_000.0)
        mock_ex.amount_to_precision.assert_called_once()
        # precision-rounded qty is passed to create_market_order
        call_qty = mock_ex.create_market_order.call_args[0][2]
        assert call_qty == pytest.approx(0.002)

    def test_paper_position_closed_as_paper_when_live_toggled(
        self, executor, temp_db, monkeypatch
    ):
        """Flipping CRYPTO_LIVE=true must not send a live sell for a paper position."""
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.1, 40000.0, 40000.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "BTC/USDT", 42_000.0)
        mock_ex = self._mock_exchange(monkeypatch)

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is True
        assert result.exchange_id is None
        mock_ex.create_market_order.assert_not_called()

    def test_live_position_closed_as_live_when_toggle_off(
        self, executor, temp_db, monkeypatch
    ):
        """CRYPTO_LIVE=false must not skip the live sell for a position opened live."""
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.2, 48000.0, 48000.0, 'open', 0)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        mock_ex = self._mock_exchange(monkeypatch, order_id="BN-999", avg_price=50_200.0)

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        assert result.is_paper is False
        assert result.exchange_id == "BN-999"
        mock_ex.create_market_order.assert_called_once()


# ---------------------------------------------------------------------------
# execute_once
# ---------------------------------------------------------------------------

class TestExecuteOnce:
    def test_empty_signals(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        results = execute_once([], equity=10_000.0)
        assert results == []

    def test_holds_excluded(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        sigs = [_make_signal(action="hold"), _make_signal(symbol="ETH/USDT", action="hold")]
        results = execute_once(sigs, equity=10_000.0)
        assert results == []

    def test_multiple_buys(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        _insert_price(temp_db, "ETH/USDT", 3_000.0)
        sigs = [
            _make_signal("BTC/USDT", action="buy", signal_id=1),
            _make_signal("ETH/USDT", action="buy", signal_id=2),
        ]
        results = execute_once(sigs, equity=100_000.0)
        # Both should go through (only 2 positions, max is 3)
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert symbols == {"BTC/USDT", "ETH/USDT"}

    def test_accepts_custom_executor(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        custom_ex = OrderExecutor()
        results = execute_once([_make_signal(action="buy")], equity=10_000.0, executor=custom_ex)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Fee and slippage (paper mode)
# ---------------------------------------------------------------------------

class TestFeeAndSlippage:
    def test_paper_buy_price_inflated_by_fee_and_slippage(
        self, executor, temp_db, monkeypatch
    ):
        """Paper buy execution price must include slippage + fee markup."""
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "FEE_PCT", 0.001)
        monkeypatch.setattr(config, "SLIPPAGE_PCT", 0.0005)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)

        assert result is not None
        expected = 50_000.0 * 1.0005 * 1.001
        assert result.price == pytest.approx(expected)
        assert result.price > 50_000.0

    def test_paper_sell_price_discounted_by_fee_and_slippage(
        self, executor, temp_db, monkeypatch
    ):
        """Paper sell execution price must be reduced by slippage + fee."""
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "FEE_PCT", 0.001)
        monkeypatch.setattr(config, "SLIPPAGE_PCT", 0.0005)

        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO positions
                   (symbol, asset_class, side, quantity, entry_price,
                    current_price, status, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.1, 50000.0, 50000.0, 'open', 1)"""
        )
        conn.commit()
        conn.close()
        _insert_price(temp_db, "BTC/USDT", 55_000.0)

        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)

        assert result is not None
        expected = 55_000.0 * 0.9995 * 0.999
        assert result.price == pytest.approx(expected)
        assert result.price < 55_000.0

    def test_round_trip_fee_reduces_pnl(self, executor, temp_db, monkeypatch):
        """A round-trip in paper mode must yield less P&L than a cost-free trade."""
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "FEE_PCT", 0.001)
        monkeypatch.setattr(config, "SLIPPAGE_PCT", 0.0005)

        # Insert two prices at distinct timestamps to avoid the UNIQUE constraint
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO prices (symbol, asset_class, timeframe, ts, open, high, low, close, volume)"
            " VALUES ('BTC/USDT', 'crypto', '1h', 1000, 50000, 50000, 50000, 50000, 1.0)"
        )
        conn.commit()
        conn.close()
        executor.execute(_make_signal(action="buy"), equity=10_000.0)

        conn = sqlite3.connect(temp_db)
        pos = conn.execute(
            "SELECT entry_price, quantity FROM positions WHERE status='open'"
        ).fetchone()
        entry_price = pos[0]  # entry_price
        quantity = pos[1]     # quantity
        conn.close()

        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO prices (symbol, asset_class, timeframe, ts, open, high, low, close, volume)"
            " VALUES ('BTC/USDT', 'crypto', '1h', 2000, 55000, 55000, 55000, 55000, 1.0)"
        )
        conn.commit()
        conn.close()
        result = executor.execute(_make_signal(action="sell"), equity=10_000.0)
        assert result is not None

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT realized_pnl FROM positions WHERE status='closed'"
        ).fetchone()
        conn.close()
        actual_pnl = row[0]  # realized_pnl

        # P&L without any costs would be (55000 - entry_price) * quantity
        # but sell price is discounted, so actual P&L must be less
        gross_pnl = (55_000.0 - entry_price) * quantity
        assert actual_pnl < gross_pnl

    def test_live_buy_does_not_apply_paper_costs(
        self, executor, temp_db, monkeypatch
    ):
        """Live orders use the exchange fill price — no extra slippage/fee layer."""
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        _insert_price(temp_db, "BTC/USDT", 50_000.0)
        mock_ex = MagicMock()
        mock_ex.amount_to_precision.side_effect = lambda sym, qty: qty
        mock_ex.create_market_order.return_value = {
            "id": "live-1",
            "average": 50_050.0,
            "price": 50_050.0,
        }
        monkeypatch.setattr("engine.order_executor._make_exchange", lambda: mock_ex)

        result = executor.execute(_make_signal(action="buy"), equity=10_000.0)

        assert result is not None
        # price must be the exchange fill price, not raw * (1+fee) * (1+slippage)
        assert result.price == pytest.approx(50_050.0)
