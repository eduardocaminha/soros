"""Tests for main.py — the bot's main loop."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestCurrentEquity:
    def test_returns_default_when_no_history(self):
        from main import _current_equity
        equity = _current_equity(is_paper=True)
        assert equity == pytest.approx(10_000.0)

    def test_returns_last_equity_snapshot(self, temp_db):
        import database.db as db_module
        conn = db_module.get_connection()
        conn.execute(
            "INSERT INTO equity_curve (equity, peak_equity, drawdown_pct, is_paper) VALUES (?, ?, ?, ?)",
            (12_000.0, 12_000.0, 0.0, 1),
        )
        conn.commit()

        from main import _current_equity
        equity = _current_equity(is_paper=True)
        assert equity == pytest.approx(12_000.0)


class TestRunCycle:
    def test_cycle_runs_without_errors(self):
        """Full cycle with all external calls mocked — should complete cleanly."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()

        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
        ):
            _run_cycle(rm)  # must not raise

    def test_cycle_skips_sentiment_when_disabled(self):
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()

        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.config.SENTIMENT_ENABLED", False),
            patch("main.sentiment_runner.run") as mock_sentiment,
        ):
            _run_cycle(rm)
            mock_sentiment.assert_not_called()

    def test_cycle_runs_sentiment_when_enabled(self):
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()

        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", return_value=[]) as mock_sentiment,
        ):
            _run_cycle(rm)
            mock_sentiment.assert_called_once()

    def test_cycle_continues_on_sentiment_failure(self):
        """Sentiment errors must not crash the cycle."""
        from engine.risk_manager import RiskManager
        from main import _run_cycle

        rm = RiskManager()

        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", side_effect=RuntimeError("rate limit")),
        ):
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

        with (
            patch("main.crypto_collector.collect_once", return_value={}),
            patch("main.stocks_collector.collect_once", return_value={}),
            patch("main.signals_compute.compute_once", return_value=[fake_signal]),
            patch("main.signal_aggregator.aggregate_once", return_value=[]),
            patch("main.order_executor.execute_once", return_value=[]),
            patch("main.stocks_executor.execute_stocks_once", return_value=[]),
            patch("main.config.SENTIMENT_ENABLED", True),
            patch("main.sentiment_runner.run", return_value=[]) as mock_sentiment,
        ):
            _run_cycle(rm)
            _, kwargs = mock_sentiment.call_args
            assert kwargs["deterministic_scores"] == {"BTC/USDT": pytest.approx(0.4)}


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
