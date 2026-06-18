"""Tests for engine/signal_aggregator.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import config
from engine.signal_aggregator import (
    AggregatedSignal,
    _action,
    _final_composite,
    aggregate_once,
    aggregate_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path: Path):
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

    orig = db_module._db

    class _FakeDB:
        def connect(self):
            c = sqlite3.connect(temp_db)
            c.row_factory = sqlite3.Row
            return c

    db_module._db = _FakeDB()
    yield
    db_module._db = orig


def _insert_signal(
    db_path: str,
    symbol: str,
    asset_class: str,
    *,
    mom: float = 0.3,
    vol: float = 0.2,
    fund: float | None = 0.1,
    composite: float = 0.2,
    action: str = "hold",
    ts: int | None = None,
) -> int:
    """Insert a signal row and return its id."""
    ts = ts or int(time.time())
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """
        INSERT INTO signals
            (symbol, asset_class, ts, momentum_score, volatility_score,
             funding_score, composite_score, action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, asset_class, ts, mom, vol, fund, composite, action),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _insert_sentiment(db_path: str, symbol: str, score: float, ts: int) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO sentiment_signals (symbol, asset_class, ts, score, label, confidence)
        VALUES (?, 'crypto', ?, ?, 'bullish', 0.8)
        """,
        (symbol, ts, score),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Unit: _final_composite
# ---------------------------------------------------------------------------

class TestFinalComposite:
    def test_crypto_positive_scores(self):
        score = _final_composite(0.5, 0.5, 0.5, 0.5, "crypto")
        assert score > 0

    def test_crypto_all_negative(self):
        score = _final_composite(-0.5, -0.5, -0.5, -0.5, "crypto")
        assert score < 0

    def test_stocks_ignores_funding(self):
        # fund param is None for stocks, only mom/vol/sent used
        score = _final_composite(0.5, 0.5, None, 0.5, "stocks")
        assert score > 0

    def test_zero_sentiment_reduces_composite(self):
        # With full sentiment weight, zero sentiment pulls composite down vs. pure deterministic
        full = _final_composite(1.0, 1.0, 1.0, 1.0, "crypto")
        reduced = _final_composite(1.0, 1.0, 1.0, 0.0, "crypto")
        assert reduced < full

    def test_output_clamped(self):
        score = _final_composite(1.0, 1.0, 1.0, 1.0, "crypto")
        assert -1.0 <= score <= 1.0

    def test_sentiment_weight_matches_config(self):
        # Only sentiment differs → difference should reflect its weight
        w = config.CRYPTO_SIGNAL_WEIGHTS
        total = sum(w.values())
        with_sent = _final_composite(0.5, 0.5, 0.5, 1.0, "crypto")
        without_sent = _final_composite(0.5, 0.5, 0.5, 0.0, "crypto")
        delta = with_sent - without_sent
        expected = w["sentiment"] / total * 1.0
        assert abs(delta - expected) < 1e-9

    def test_missing_funding_treated_as_zero(self):
        score_with = _final_composite(0.5, 0.5, 0.0, 0.5, "crypto")
        score_none = _final_composite(0.5, 0.5, None, 0.5, "crypto")
        assert score_with == pytest.approx(score_none)


# ---------------------------------------------------------------------------
# Unit: _action
# ---------------------------------------------------------------------------

class TestAction:
    def test_above_threshold_is_buy(self):
        assert _action(config.SIGNAL_THRESHOLD + 0.01) == "buy"

    def test_below_neg_threshold_is_sell(self):
        assert _action(-(config.SIGNAL_THRESHOLD + 0.01)) == "sell"

    def test_within_threshold_is_hold(self):
        assert _action(0.0) == "hold"
        assert _action(config.SIGNAL_THRESHOLD - 0.01) == "hold"


# ---------------------------------------------------------------------------
# Integration: aggregate_signal
# ---------------------------------------------------------------------------

class TestAggregateSignal:
    def test_returns_none_when_no_signal_row(self):
        result = aggregate_signal("MISSING/USDT", "crypto")
        assert result is None

    def test_returns_aggregated_signal_with_no_sentiment(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        _insert_signal(temp_db, "BTC/USDT", "crypto")
        result = aggregate_signal("BTC/USDT", "crypto")
        assert isinstance(result, AggregatedSignal)
        assert result.symbol == "BTC/USDT"
        assert result.asset_class == "crypto"
        assert result.sentiment_score == 0.0
        assert -1.0 <= result.composite_score <= 1.0
        assert result.action in ("buy", "sell", "hold")

    def test_blends_sentiment_for_crypto(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_signal(temp_db, "BTC/USDT", "crypto", mom=0.0, vol=0.0, fund=0.0, composite=0.0)
        _insert_sentiment(temp_db, "BTC/USDT", 1.0, now - 60)
        result = aggregate_signal("BTC/USDT", "crypto")
        assert result is not None
        # Sentiment=1.0, everything else=0 → composite = sent_weight * 1.0 / total
        w = config.CRYPTO_SIGNAL_WEIGHTS
        expected = w["sentiment"] / sum(w.values())
        assert result.composite_score == pytest.approx(expected)

    def test_blends_sentiment_for_stocks(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_signal(temp_db, "AAPL", "stocks", fund=None, mom=0.0, vol=0.0, composite=0.0)
        _insert_sentiment(temp_db, "AAPL", 1.0, now - 60)
        result = aggregate_signal("AAPL", "stocks")
        assert result is not None
        w = config.STOCK_SIGNAL_WEIGHTS
        expected = w["sentiment"] / sum(w.values())
        assert result.composite_score == pytest.approx(expected)

    def test_updates_db_composite_and_action(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        row_id = _insert_signal(
            temp_db, "ETH/USDT", "crypto", mom=1.0, vol=1.0, fund=1.0, composite=0.0
        )
        _insert_sentiment(temp_db, "ETH/USDT", 1.0, now - 60)
        result = aggregate_signal("ETH/USDT", "crypto")
        assert result is not None

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT composite_score, action FROM signals WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()

        assert row[0] == pytest.approx(result.composite_score)
        assert row[1] == result.action

    def test_picks_latest_signal_row(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        now = int(time.time())
        _insert_signal(temp_db, "BTC/USDT", "crypto", mom=0.1, ts=now - 3600)
        _insert_signal(temp_db, "BTC/USDT", "crypto", mom=0.9, ts=now)
        result = aggregate_signal("BTC/USDT", "crypto")
        assert result is not None
        assert result.momentum_score == pytest.approx(0.9)

    def test_positive_sentiment_can_push_to_buy(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        # All signals at max positive → should be a buy
        _insert_signal(temp_db, "BTC/USDT", "crypto", mom=1.0, vol=1.0, fund=1.0, composite=0.0)
        _insert_sentiment(temp_db, "BTC/USDT", 1.0, now - 60)
        result = aggregate_signal("BTC/USDT", "crypto")
        assert result is not None
        assert result.action == "buy"

    def test_negative_sentiment_can_push_to_sell(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_signal(temp_db, "BTC/USDT", "crypto", mom=-1.0, vol=-1.0, fund=-1.0, composite=0.0)
        _insert_sentiment(temp_db, "BTC/USDT", -1.0, now - 60)
        result = aggregate_signal("BTC/USDT", "crypto")
        assert result is not None
        assert result.action == "sell"

    def test_stocks_funding_score_is_none(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        _insert_signal(temp_db, "AAPL", "stocks", fund=None)
        result = aggregate_signal("AAPL", "stocks")
        assert result is not None
        assert result.funding_score is None


# ---------------------------------------------------------------------------
# Integration: aggregate_once
# ---------------------------------------------------------------------------

class TestAggregateOnce:
    def test_returns_empty_when_no_signals(self, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        results = aggregate_once(["BTC/USDT"], ["AAPL"])
        assert results == []

    def test_processes_crypto_and_stocks(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        _insert_signal(temp_db, "BTC/USDT", "crypto")
        _insert_signal(temp_db, "AAPL", "stocks", fund=None)
        results = aggregate_once(["BTC/USDT"], ["AAPL"])
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert symbols == {"BTC/USDT", "AAPL"}

    def test_skips_missing_symbols(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        _insert_signal(temp_db, "BTC/USDT", "crypto")
        results = aggregate_once(["BTC/USDT", "ETH/USDT"], [])
        assert len(results) == 1
        assert results[0].symbol == "BTC/USDT"
