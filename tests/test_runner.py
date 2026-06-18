"""Tests for sentiment/runner.py."""

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from sentiment.runner import SentimentRecord, _score_to_label, run


# ---------------------------------------------------------------------------
# _score_to_label
# ---------------------------------------------------------------------------

class TestScoreToLabel:
    def test_bullish(self):
        assert _score_to_label(0.5) == "bullish"

    def test_bearish(self):
        assert _score_to_label(-0.5) == "bearish"

    def test_neutral_positive_edge(self):
        assert _score_to_label(0.05) == "neutral"

    def test_neutral_negative_edge(self):
        assert _score_to_label(-0.05) == "neutral"

    def test_neutral_zero(self):
        assert _score_to_label(0.0) == "neutral"

    def test_boundary_bullish(self):
        # 0.1 is the boundary — just above is bullish
        assert _score_to_label(0.11) == "bullish"

    def test_boundary_bearish(self):
        assert _score_to_label(-0.11) == "bearish"


# ---------------------------------------------------------------------------
# run() — integration via mocked dependencies
# ---------------------------------------------------------------------------

def _make_debate_result(symbol: str, score: float = 0.4, debated: bool = False):
    from sentiment.debate import DebateResult
    return DebateResult(
        symbol=symbol,
        score=score,
        rationale="test rationale",
        debated=debated,
        debated_at=int(time.time()),
    )


def _in_memory_conn():
    """Return an in-memory SQLite connection with the sentinel_signals table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            score       REAL NOT NULL,
            label       TEXT NOT NULL,
            confidence  REAL NOT NULL,
            debate_used INTEGER NOT NULL DEFAULT 0,
            raw_json    TEXT,
            inserted_at INTEGER NOT NULL DEFAULT (unixepoch())
        )
    """)
    conn.commit()
    return conn


@pytest.fixture()
def mock_deps(monkeypatch):
    """Patch _analyse_symbol and get_connection so run() uses in-memory DB."""
    conn = _in_memory_conn()
    monkeypatch.setattr("sentiment.runner.get_connection", lambda: conn)
    return conn


class TestRun:
    def test_returns_records_for_each_symbol(self, mock_deps, monkeypatch):
        def fake_analyse(symbol, asset_class, client, det_score):
            return _make_debate_result(symbol, score=0.5)

        monkeypatch.setattr("sentiment.runner._analyse_symbol", fake_analyse)

        records = run(
            crypto_symbols=["BTC/USDT", "ETH/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        assert len(records) == 2
        assert {r.symbol for r in records} == {"BTC/USDT", "ETH/USDT"}

    def test_record_fields_populated(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: _make_debate_result("BTC/USDT", score=0.6, debated=True),
        )
        records = run(
            crypto_symbols=["BTC/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        r = records[0]
        assert r.symbol == "BTC/USDT"
        assert r.asset_class == "crypto"
        assert r.score == pytest.approx(0.6)
        assert r.label == "bullish"
        assert r.confidence == pytest.approx(0.6)
        assert r.debate_used is True
        assert r.raw_json is not None

    def test_record_persisted_to_db(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: _make_debate_result("ETH/USDT", score=-0.3),
        )
        run(
            crypto_symbols=["ETH/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        row = mock_deps.execute(
            "SELECT * FROM sentiment_signals WHERE symbol = 'ETH/USDT'"
        ).fetchone()
        assert row is not None
        assert row["score"] == pytest.approx(-0.3)
        assert row["label"] == "bearish"
        assert row["asset_class"] == "crypto"

    def test_stock_symbols_use_stocks_asset_class(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda symbol, *args, **kwargs: _make_debate_result(symbol, score=0.2),
        )
        run(
            crypto_symbols=[],
            stock_symbols=["AAPL"],
            client=MagicMock(),
        )
        row = mock_deps.execute(
            "SELECT asset_class FROM sentiment_signals WHERE symbol = 'AAPL'"
        ).fetchone()
        assert row["asset_class"] == "stocks"

    def test_deterministic_score_passed_to_analyse(self, mock_deps, monkeypatch):
        received: list[float] = []

        def fake_analyse(symbol, asset_class, client, det_score):
            received.append(det_score)
            return _make_debate_result(symbol)

        monkeypatch.setattr("sentiment.runner._analyse_symbol", fake_analyse)

        run(
            crypto_symbols=["BTC/USDT"],
            stock_symbols=[],
            deterministic_scores={"BTC/USDT": 0.75},
            client=MagicMock(),
        )
        assert received == [pytest.approx(0.75)]

    def test_missing_deterministic_score_defaults_to_zero(self, mock_deps, monkeypatch):
        received: list[float] = []

        def fake_analyse(symbol, asset_class, client, det_score):
            received.append(det_score)
            return _make_debate_result(symbol)

        monkeypatch.setattr("sentiment.runner._analyse_symbol", fake_analyse)

        run(
            crypto_symbols=["BTC/USDT"],
            stock_symbols=[],
            deterministic_scores={},
            client=MagicMock(),
        )
        assert received == [pytest.approx(0.0)]

    def test_error_in_one_symbol_does_not_abort_others(self, mock_deps, monkeypatch):
        call_count = 0

        def fake_analyse(symbol, asset_class, client, det_score):
            nonlocal call_count
            call_count += 1
            if symbol == "BTC/USDT":
                raise RuntimeError("network error")
            return _make_debate_result(symbol, score=0.3)

        monkeypatch.setattr("sentiment.runner._analyse_symbol", fake_analyse)

        records = run(
            crypto_symbols=["BTC/USDT", "ETH/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        assert call_count == 2
        assert len(records) == 1
        assert records[0].symbol == "ETH/USDT"

    def test_raw_json_is_valid_json(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: _make_debate_result("BTC/USDT", score=0.4, debated=False),
        )
        records = run(
            crypto_symbols=["BTC/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        raw = json.loads(records[0].raw_json)
        assert "score" in raw
        assert "rationale" in raw
        assert "debated" in raw

    def test_confidence_equals_abs_score(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: _make_debate_result("BTC/USDT", score=-0.75),
        )
        records = run(
            crypto_symbols=["BTC/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        assert records[0].confidence == pytest.approx(0.75)

    def test_neutral_label_on_near_zero_score(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: _make_debate_result("BTC/USDT", score=0.05),
        )
        records = run(
            crypto_symbols=["BTC/USDT"],
            stock_symbols=[],
            client=MagicMock(),
        )
        assert records[0].label == "neutral"

    def test_empty_symbols_returns_empty_list(self, mock_deps, monkeypatch):
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        records = run(crypto_symbols=[], stock_symbols=[], client=MagicMock())
        assert records == []

    def test_client_created_when_not_supplied(self, mock_deps, monkeypatch):
        created: list[object] = []

        class FakeClient:
            def __init__(self):
                created.append(self)

        monkeypatch.setattr("sentiment.runner.ClaudeClient", FakeClient)
        monkeypatch.setattr(
            "sentiment.runner._analyse_symbol",
            lambda *_: _make_debate_result("BTC/USDT"),
        )
        run(crypto_symbols=["BTC/USDT"], stock_symbols=[])
        assert len(created) == 1
