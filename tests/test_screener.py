"""Tests for engine/screener.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import config
from engine.screener import (
    ScreenerEntry,
    ScreenerResult,
    _SENTIMENT_GATE,
    _latest_composite,
    _latest_sentiment,
    _volume_usd_24h,
    screen,
)


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
    orig = db_module._db

    class _FakeDB:
        def connect(self):
            c = sqlite3.connect(temp_db)
            c.row_factory = sqlite3.Row
            return c

    db_module._db = _FakeDB()
    yield
    db_module._db = orig


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _insert_prices(
    db_path: str,
    symbol: str,
    asset_class: str,
    n: int = 24,
    close: float = 100.0,
    volume: float = 1_000.0,
) -> None:
    conn = sqlite3.connect(db_path)
    now = int(time.time())
    for i in range(n):
        conn.execute(
            """
            INSERT OR IGNORE INTO prices
                (symbol, asset_class, timeframe, ts, open, high, low, close, volume)
            VALUES (?, ?, '1h', ?, ?, ?, ?, ?, ?)
            """,
            (symbol, asset_class, now - (n - i) * 3600,
             close, close * 1.01, close * 0.99, close, volume),
        )
    conn.commit()
    conn.close()


def _insert_signal(
    db_path: str,
    symbol: str,
    asset_class: str,
    composite: float = 0.5,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO signals
            (symbol, asset_class, ts, momentum_score, volatility_score,
             composite_score, action)
        VALUES (?, ?, ?, 0.3, 0.2, ?, 'buy')
        """,
        (symbol, asset_class, int(time.time()), composite),
    )
    conn.commit()
    conn.close()


def _insert_sentiment(
    db_path: str,
    symbol: str,
    asset_class: str,
    score: float = 0.5,
    age_seconds: int = 60,
) -> None:
    conn = sqlite3.connect(db_path)
    label = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")
    conn.execute(
        """
        INSERT INTO sentiment_signals
            (symbol, asset_class, ts, score, label, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (symbol, asset_class, int(time.time()) - age_seconds, score, label, abs(score)),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Unit: DB helpers
# ---------------------------------------------------------------------------

class TestVolumeUsd24h:
    def test_returns_zero_when_no_data(self):
        assert _volume_usd_24h("MISSING/USDT", "crypto") == 0.0

    def test_sums_close_times_volume(self, temp_db):
        _insert_prices(temp_db, "BTC/USDT", "crypto", n=24, close=50_000.0, volume=2.0)
        vol = _volume_usd_24h("BTC/USDT", "crypto")
        assert vol == pytest.approx(24 * 50_000.0 * 2.0)

    def test_uses_only_last_24_candles(self, temp_db):
        _insert_prices(temp_db, "ETH/USDT", "crypto", n=48, close=3_000.0, volume=10.0)
        vol = _volume_usd_24h("ETH/USDT", "crypto")
        assert vol == pytest.approx(24 * 3_000.0 * 10.0)


class TestLatestComposite:
    def test_returns_zero_when_absent(self):
        assert _latest_composite("MISSING", "stocks") == 0.0

    def test_returns_composite_score(self, temp_db):
        _insert_signal(temp_db, "AAPL", "stocks", composite=0.7)
        assert _latest_composite("AAPL", "stocks") == pytest.approx(0.7)

    def test_returns_latest_when_multiple_rows(self, temp_db):
        conn = sqlite3.connect(temp_db)
        now = int(time.time())
        for ts, composite in [(now - 3600, 0.1), (now, 0.9)]:
            conn.execute(
                """
                INSERT INTO signals
                    (symbol, asset_class, ts, momentum_score, volatility_score,
                     composite_score, action)
                VALUES ('BTC/USDT', 'crypto', ?, 0.1, 0.1, ?, 'buy')
                """,
                (ts, composite),
            )
        conn.commit()
        conn.close()
        assert _latest_composite("BTC/USDT", "crypto") == pytest.approx(0.9)


class TestLatestSentiment:
    def test_returns_zero_when_absent(self):
        assert _latest_sentiment("MISSING", "crypto") == 0.0

    def test_returns_score_when_fresh(self, temp_db):
        _insert_sentiment(temp_db, "BTC/USDT", "crypto", score=0.8, age_seconds=60)
        assert _latest_sentiment("BTC/USDT", "crypto") == pytest.approx(0.8)

    def test_returns_zero_when_stale(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_MAX_AGE_SECONDS", 3600)
        _insert_sentiment(temp_db, "BTC/USDT", "crypto", score=0.8, age_seconds=7200)
        assert _latest_sentiment("BTC/USDT", "crypto") == 0.0

    def test_negative_score_returned(self, temp_db):
        _insert_sentiment(temp_db, "ETH/USDT", "crypto", score=-0.6, age_seconds=10)
        assert _latest_sentiment("ETH/USDT", "crypto") == pytest.approx(-0.6)


# ---------------------------------------------------------------------------
# Integration: screen() — SCREENER_ENABLED=False (default)
# ---------------------------------------------------------------------------

class TestScreenDisabled:
    def test_returns_only_pinned_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", False)
        result = screen(
            crypto_pinned=["BTC/USDT", "ETH/USDT"],
            crypto_watchlist=["DOGE/USDT"],
            stock_pinned=["AAPL"],
            stock_watchlist=["TSLA"],
        )
        assert result.selected_crypto == ["BTC/USDT", "ETH/USDT"]
        assert result.selected_stocks == ["AAPL"]

    def test_entries_include_pinned_only_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", False)
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=["DOGE/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        syms = [e.symbol for e in result.entries]
        assert "BTC/USDT" in syms
        assert "DOGE/USDT" not in syms

    def test_pinned_entries_have_reason_pinned(self, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", False)
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=[],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert result.entries[0].reason == "pinned"
        assert result.entries[0].selected is True

    def test_returns_screener_result_type(self, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", False)
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=[],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert isinstance(result, ScreenerResult)
        assert all(isinstance(e, ScreenerEntry) for e in result.entries)

    def test_uses_config_defaults_when_no_args(self, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", False)
        monkeypatch.setattr(config, "CRYPTO_SYMBOLS", ["BTC/USDT"])
        monkeypatch.setattr(config, "STOCK_SYMBOLS", ["AAPL"])
        monkeypatch.setattr(config, "CRYPTO_WATCHLIST", [])
        monkeypatch.setattr(config, "STOCK_WATCHLIST", [])
        result = screen()
        assert "BTC/USDT" in result.selected_crypto
        assert "AAPL" in result.selected_stocks


# ---------------------------------------------------------------------------
# Integration: screen() — SCREENER_ENABLED=True
# ---------------------------------------------------------------------------

class TestScreenEnabled:
    def test_pinned_always_selected(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000_000.0)
        # Do NOT insert prices for pinned → volume=0; still selected
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=[],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert "BTC/USDT" in result.selected_crypto

    def test_watchlist_passes_volume_floor(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 3)
        # ETH/USDT: volume = 24 * 3000 * 20 = 1_440_000 (passes)
        _insert_prices(temp_db, "ETH/USDT", "crypto", close=3_000.0, volume=20.0)
        _insert_signal(temp_db, "ETH/USDT", "crypto", composite=0.6)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["ETH/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert "ETH/USDT" in result.selected_crypto

    def test_watchlist_rejected_below_volume_floor(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 5_000_000.0)
        # volume = 24 * 100 * 10 = 24_000 (below floor)
        _insert_prices(temp_db, "DOGE/USDT", "crypto", close=100.0, volume=10.0)
        _insert_signal(temp_db, "DOGE/USDT", "crypto", composite=0.9)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["DOGE/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert "DOGE/USDT" not in result.selected_crypto
        entry = next(e for e in result.entries if e.symbol == "DOGE/USDT")
        assert entry.reason == "volume_floor"

    def test_watchlist_rejected_by_sentiment_gate(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 3)
        _insert_prices(temp_db, "ADA/USDT", "crypto", close=1.0, volume=10_000.0)
        _insert_signal(temp_db, "ADA/USDT", "crypto", composite=0.8)
        _insert_sentiment(temp_db, "ADA/USDT", "crypto", score=_SENTIMENT_GATE - 0.01)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["ADA/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert "ADA/USDT" not in result.selected_crypto
        entry = next(e for e in result.entries if e.symbol == "ADA/USDT")
        assert entry.reason == "sentiment_gate"

    def test_neutral_sentiment_passes_gate(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 3)
        _insert_prices(temp_db, "SOL/USDT", "crypto", close=100.0, volume=1_000.0)
        _insert_signal(temp_db, "SOL/USDT", "crypto", composite=0.5)
        # No sentiment row → defaults to 0.0 → passes gate
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["SOL/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert "SOL/USDT" in result.selected_crypto

    def test_top_n_limits_watchlist_additions(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 2)
        monkeypatch.setattr(config, "MAX_OPEN_POSITIONS", 10)  # not the cap here
        symbols = ["SYM1/USDT", "SYM2/USDT", "SYM3/USDT", "SYM4/USDT"]
        for i, sym in enumerate(symbols):
            _insert_prices(temp_db, sym, "crypto", close=100.0, volume=1_000.0)
            _insert_signal(temp_db, sym, "crypto", composite=0.5 + i * 0.1)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=symbols,
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert len(result.selected_crypto) == 2

    def test_ranks_by_conviction_descending(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 1)
        monkeypatch.setattr(config, "MAX_OPEN_POSITIONS", 5)
        # HIGH conviction
        _insert_prices(temp_db, "HIGH/USDT", "crypto", close=100.0, volume=1_000.0)
        _insert_signal(temp_db, "HIGH/USDT", "crypto", composite=0.9)
        # LOW conviction
        _insert_prices(temp_db, "LOW/USDT", "crypto", close=100.0, volume=1_000.0)
        _insert_signal(temp_db, "LOW/USDT", "crypto", composite=0.1)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["HIGH/USDT", "LOW/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert result.selected_crypto == ["HIGH/USDT"]

    def test_sentiment_as_tiebreaker(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 1)
        monkeypatch.setattr(config, "MAX_OPEN_POSITIONS", 5)
        # Same conviction, different sentiment
        for sym, sent in [("A/USDT", 0.8), ("B/USDT", 0.2)]:
            _insert_prices(temp_db, sym, "crypto", close=100.0, volume=1_000.0)
            _insert_signal(temp_db, sym, "crypto", composite=0.5)
            _insert_sentiment(temp_db, sym, "crypto", score=sent)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["A/USDT", "B/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert result.selected_crypto == ["A/USDT"]

    def test_pinned_in_watchlist_not_duplicated(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        _insert_prices(temp_db, "BTC/USDT", "crypto", close=50_000.0, volume=10.0)
        _insert_signal(temp_db, "BTC/USDT", "crypto", composite=0.7)
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=["BTC/USDT", "ETH/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert result.selected_crypto.count("BTC/USDT") == 1

    def test_negative_conviction_still_ranks(self, temp_db, monkeypatch):
        """Conviction = |composite| so bearish signals can still qualify."""
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 1)
        monkeypatch.setattr(config, "MAX_OPEN_POSITIONS", 5)
        _insert_prices(temp_db, "BEAR/USDT", "crypto", close=100.0, volume=1_000.0)
        _insert_signal(temp_db, "BEAR/USDT", "crypto", composite=-0.8)
        # Sentiment is neutral (passes gate)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=["BEAR/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert "BEAR/USDT" in result.selected_crypto

    def test_top_n_capped_at_max_open_positions(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 10)
        monkeypatch.setattr(config, "MAX_OPEN_POSITIONS", 2)
        for i in range(5):
            sym = f"SYM{i}/USDT"
            _insert_prices(temp_db, sym, "crypto", close=100.0, volume=1_000.0)
            _insert_signal(temp_db, sym, "crypto", composite=0.5)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=[f"SYM{i}/USDT" for i in range(5)],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert len(result.selected_crypto) <= 2

    def test_stocks_screened_independently(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        monkeypatch.setattr(config, "SCREENER_TOP_N", 3)
        _insert_prices(temp_db, "TSLA", "stocks", close=200.0, volume=100.0)
        _insert_signal(temp_db, "TSLA", "stocks", composite=0.6)
        result = screen(
            crypto_pinned=[],
            crypto_watchlist=[],
            stock_pinned=["AAPL"],
            stock_watchlist=["TSLA"],
        )
        assert "AAPL" in result.selected_stocks
        assert "TSLA" in result.selected_stocks

    def test_entries_cover_full_universe(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        monkeypatch.setattr(config, "SCREENER_MIN_VOLUME_USD", 1_000.0)
        _insert_prices(temp_db, "DOGE/USDT", "crypto", close=0.1, volume=10_000.0)
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=["DOGE/USDT"],
            stock_pinned=[],
            stock_watchlist=[],
        )
        syms = {e.symbol for e in result.entries}
        assert "BTC/USDT" in syms
        assert "DOGE/USDT" in syms

    def test_empty_watchlist_returns_only_pinned(self, monkeypatch):
        monkeypatch.setattr(config, "SCREENER_ENABLED", True)
        result = screen(
            crypto_pinned=["BTC/USDT"],
            crypto_watchlist=[],
            stock_pinned=[],
            stock_watchlist=[],
        )
        assert result.selected_crypto == ["BTC/USDT"]
