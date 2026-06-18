"""Tests for signals/sentiment.py — the 4th signal that reads from SQLite."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import config
from signals.sentiment import compute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path: Path):
    """Initialise a temp SQLite DB with the full schema."""
    db_file = str(tmp_path / "test.db")
    schema = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()
    conn = sqlite3.connect(db_file)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    return db_file


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


@pytest.fixture(autouse=True)
def _patch_db(temp_db: str, monkeypatch):
    """Point database.db at the temp DB for every test."""
    import database.db as db_module

    orig = db_module._db

    class _FakeDB:
        def connect(self):
            c = sqlite3.connect(temp_db)
            c.row_factory = sqlite3.Row
            return c

    db_module._db = _FakeDB()
    # Also reset the cached connection so compute() picks up the fake DB.
    db_module._conn = None
    yield
    db_module._db = orig
    db_module._conn = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeSentimentSignal:
    def test_returns_zero_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        assert compute("BTC/USDT") == 0.0

    def test_returns_zero_when_no_row(self, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        assert compute("BTC/USDT") == 0.0

    def test_returns_score_for_fresh_row(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_sentiment(temp_db, "BTC/USDT", 0.75, now - 60)
        score = compute("BTC/USDT", now=now)
        assert score == pytest.approx(0.75)

    def test_returns_zero_for_stale_row(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        stale_ts = now - config.SENTIMENT_MAX_AGE_SECONDS - 1
        _insert_sentiment(temp_db, "BTC/USDT", 0.9, stale_ts)
        assert compute("BTC/USDT", now=now) == 0.0

    def test_picks_latest_row(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_sentiment(temp_db, "BTC/USDT", 0.5, now - 3600)
        _insert_sentiment(temp_db, "BTC/USDT", -0.3, now - 60)
        score = compute("BTC/USDT", now=now)
        assert score == pytest.approx(-0.3)

    def test_symbol_isolation(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_sentiment(temp_db, "ETH/USDT", 0.6, now - 60)
        assert compute("BTC/USDT", now=now) == 0.0

    def test_score_at_positive_boundary(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_sentiment(temp_db, "BTC/USDT", 1.0, now - 60)
        assert compute("BTC/USDT", now=now) == pytest.approx(1.0)

    def test_score_at_negative_boundary(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        _insert_sentiment(temp_db, "BTC/USDT", -1.0, now - 60)
        assert compute("BTC/USDT", now=now) == pytest.approx(-1.0)

    def test_exactly_at_max_age_is_fresh(self, temp_db: str, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        now = int(time.time())
        ts = now - config.SENTIMENT_MAX_AGE_SECONDS
        _insert_sentiment(temp_db, "BTC/USDT", 0.4, ts)
        assert compute("BTC/USDT", now=now) == pytest.approx(0.4)
