"""Tests for main.py — the bot's main loop."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


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
            c.execute("PRAGMA journal_mode=WAL")
            return c

    monkeypatch.setattr(db_module, "_db", _FakeDB())


def _make_aggregated(symbol="BTC/USDT", asset_class="crypto", action="hold"):
    return SimpleNamespace(
        symbol=symbol,
        asset_class=asset_class,
        signal_id=1,
        momentum_score=0.1,
        volatility_score=0.1,
        funding_score=0.0,
        sentiment_score=0.0,
        composite_score=0.1,
        action=action,
    )


def _make_screener_result(crypto=None, stocks=None):
    from engine.screener import ScreenerResult
    return ScreenerResult(
        selected_crypto=crypto or ["BTC/USDT"],
        selected_stocks=stocks or ["AAPL"],
        entries=[],
    )


class TestCurrentEquity:
    def test_returns_initial_capital_when_no_positions(self):
        import config
        from main import _current_equity
        equity = _current_equity()
        assert equity == pytest.approx(config.INITIAL_CAPITAL)

    def test_equity_includes_unrealized_pnl(self, temp_db):
        import config
        import database.db as db_module
        conn = db_module.get_connection()
        conn.execute(
            """INSERT INTO positions
               (symbol, asset_class, side, quantity, entry_price,
                current_price, unrealized_pnl, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.1, 50000.0, 55000.0, 500.0, 1)"""
        )
        conn.commit()

        from main import _current_equity
        equity = _current_equity()
        assert equity == pytest.approx(config.INITIAL_CAPITAL + 500.0)

    def test_equity_includes_realized_pnl(self, temp_db):
        import config
        import database.db as db_module
        conn = db_module.get_connection()
        conn.execute(
            """INSERT INTO positions
               (symbol, asset_class, side, quantity, entry_price,
                current_price, realized_pnl, status, is_paper)
               VALUES ('ETH/USDT', 'crypto', 'long', 1.0, 2000.0, 2500.0, 500.0, 'closed', 1)"""
        )
        conn.commit()

        from main import _current_equity
        equity = _current_equity()
        assert equity == pytest.approx(config.INITIAL_CAPITAL + 500.0)

    def test_equity_sums_both_realized_and_unrealized(self, temp_db):
        import config
        import database.db as db_module
        conn = db_module.get_connection()
        conn.execute(
            """INSERT INTO positions
               (symbol, asset_class, side, quantity, entry_price,
                current_price, realized_pnl, status, is_paper)
               VALUES ('ETH/USDT', 'crypto', 'long', 1.0, 2000.0, 2200.0, 200.0, 'closed', 1)"""
        )
        conn.execute(
            """INSERT INTO positions
               (symbol, asset_class, side, quantity, entry_price,
                current_price, unrealized_pnl, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.05, 40000.0, 42000.0, 100.0, 1)"""
        )
        conn.commit()

        from main import _current_equity
        equity = _current_equity()
        assert equity == pytest.approx(config.INITIAL_CAPITAL + 200.0 + 100.0)


class TestMarkToMarket:
    def test_updates_current_price_and_unrealized_pnl(self, temp_db):
        import database.db as db_module
        from main import _mark_to_market

        conn = db_module.get_connection()
        conn.execute(
            """INSERT INTO positions
               (symbol, asset_class, side, quantity, entry_price, current_price, is_paper)
               VALUES ('BTC/USDT', 'crypto', 'long', 0.1, 50000.0, 50000.0, 1)"""
        )
        conn.execute(
            """INSERT INTO prices
               (symbol, asset_class, timeframe, ts, open, high, low, close, volume)
               VALUES ('BTC/USDT', 'crypto', '1h', 1000, 55000, 55000, 55000, 55000, 1.0)"""
        )
        conn.commit()

        _mark_to_market()

        row = conn.execute(
            "SELECT current_price, unrealized_pnl FROM positions WHERE status = 'open'"
        ).fetchone()
        assert row["current_price"] == pytest.approx(55_000.0)
        assert row["unrealized_pnl"] == pytest.approx(0.1 * (55_000.0 - 50_000.0))

    def test_skips_position_with_no_price(self, temp_db):
        import database.db as db_module
        from main import _mark_to_market

        conn = db_module.get_connection()
        conn.execute(
            """INSERT INTO positions
               (symbol, asset_class, side, quantity, entry_price,
                current_price, unrealized_pnl, is_paper)
               VALUES ('SOL/USDT', 'crypto', 'long', 10.0, 100.0, 100.0, 0.0, 1)"""
        )
        conn.commit()

        _mark_to_market()  # no price row for SOL/USDT — must not raise or corrupt

        row = conn.execute(
            "SELECT current_price, unrealized_pnl FROM positions WHERE status = 'open'"
        ).fetchone()
        assert row["current_price"] == pytest.approx(100.0)
        assert row["unrealized_pnl"] == pytest.approx(0.0)


_CYCLE_BASE_PATCHES = [
    ("main.crypto_collector.collect_once", {}),
    ("main.stocks_collector.collect_once", {}),
    ("main.signals_compute.compute_once", []),
    ("main.signal_aggregator.aggregate_once", []),
    ("main.order_executor.execute_once", []),
    ("main.stocks_executor.execute_stocks_once", []),
]


def _cycle_ctx(extra=None):
    """Context-manager stack: base patches + optional extras."""
    from contextlib import ExitStack
    stack = ExitStack()
    for target, retval in _CYCLE_BASE_PATCHES:
        stack.enter_context(patch(target, return_value=retval))
    stack.enter_context(
        patch("main.screen", return_value=_make_screener_result())
    )
    for ctx in (extra or []):
        stack.enter_context(ctx)
    return stack


class TestRunCycle:
    def test_cycle_runs_without_errors(self):
        """Full cycle with all external calls mocked — should complete cleanly."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        with _cycle_ctx():
            _run_cycle(rm)  # must not raise

    def test_cycle_skips_sentiment_when_disabled(self):
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        mock_sentiment = MagicMock()
        with _cycle_ctx([
            patch("main.config.SENTIMENT_ENABLED", False),
            patch("main.sentiment_runner.run", mock_sentiment),
        ]):
            _run_cycle(rm)
            mock_sentiment.assert_not_called()

    def test_cycle_runs_sentiment_when_enabled(self):
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        mock_sentiment = MagicMock(return_value=[])
        with _cycle_ctx([
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", mock_sentiment),
        ]):
            _run_cycle(rm)
            mock_sentiment.assert_called_once()

    def test_cycle_continues_on_sentiment_failure(self):
        """Sentiment errors must not crash the cycle."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        with _cycle_ctx([
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", side_effect=RuntimeError("rate limit")),
        ]):
            _run_cycle(rm)  # must not raise

    def test_cycle_passes_det_scores_to_sentiment(self):
        """Deterministic composite scores must be forwarded to the sentiment runner."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle
        from signals.compute import SignalResult

        rm = RiskManager()
        fake_signal = SignalResult(
            symbol="BTC/USDT",
            asset_class="crypto",
            ts=1000,
            momentum_score=0.5,
            volatility_score=0.3,
            funding_score=0.1,
            composite_score=0.4,
            action="buy",
        )
        mock_sentiment = MagicMock(return_value=[])
        with _cycle_ctx([
            patch("main.signals_compute.compute_once", return_value=[fake_signal]),
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", mock_sentiment),
        ]):
            _run_cycle(rm)
            _, kwargs = mock_sentiment.call_args
            assert kwargs["deterministic_scores"] == {"BTC/USDT": pytest.approx(0.4)}

    def test_cycle_calls_screener_each_cycle(self):
        """screen() must be called once per cycle."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        mock_screen = MagicMock(return_value=_make_screener_result())
        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.screen", mock_screen),
        ):
            _run_cycle(rm)
            mock_screen.assert_called_once()

    def test_cycle_forwards_selected_symbols_to_compute(self):
        """compute_once must receive the screener-selected symbols."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        screener_result = _make_screener_result(
            crypto=["ETH/USDT"], stocks=["MSFT"]
        )
        mock_compute = MagicMock(return_value=[])
        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", mock_compute),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.screen", return_value=screener_result),
        ):
            _run_cycle(rm)
            mock_compute.assert_called_once_with(
                crypto_symbols=["ETH/USDT"],
                stock_symbols=["MSFT"],
            )

    def test_cycle_forwards_selected_symbols_to_aggregate(self):
        """aggregate_once must receive the screener-selected symbols (and origins)."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        screener_result = _make_screener_result(
            crypto=["BTC/USDT", "SOL/USDT"], stocks=["AAPL", "NVDA"]
        )
        mock_aggregate = MagicMock(return_value=[])
        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", mock_aggregate),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.screen", return_value=screener_result),
        ):
            _run_cycle(rm)
            mock_aggregate.assert_called_once()
            kwargs = mock_aggregate.call_args[1]
            assert kwargs["crypto_symbols"] == ["BTC/USDT", "SOL/USDT"]
            assert kwargs["stock_symbols"] == ["AAPL", "NVDA"]
            assert "origins" in kwargs  # origins dict is now always passed

    def test_cycle_forwards_selected_symbols_to_sentiment(self):
        """sentiment_runner.run must receive the screener-selected symbols."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()
        screener_result = _make_screener_result(
            crypto=["BTC/USDT"], stocks=["AAPL"]
        )
        mock_sentiment = MagicMock(return_value=[])
        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.screen", return_value=screener_result),
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", mock_sentiment),
        ):
            _run_cycle(rm)
            _, kwargs = mock_sentiment.call_args
            assert kwargs["crypto_symbols"] == ["BTC/USDT"]
            assert kwargs["stock_symbols"] == ["AAPL"]


class TestMain:
    def test_main_validates_config_on_startup(self):
        """main() must call validate_config before entering the loop.

        _run_cycle raises KeyboardInterrupt (a BaseException, not Exception) so it
        escapes the bare `except Exception` handler inside the while-loop and
        propagates out of main() immediately — no sleep, no hang.
        """
        import main as main_module

        with (
            patch("main.config.validate_config") as mock_validate,
            patch("main._run_cycle", side_effect=KeyboardInterrupt),
            patch("main.get_connection"),
        ):
            try:
                main_module.main()
            except (KeyboardInterrupt, SystemExit):
                pass

        mock_validate.assert_called_once()
