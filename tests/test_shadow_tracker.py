"""Tests for engine/shadow_tracker.py — forward shadow scoring."""

from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from engine.shadow_tracker import (
    ShadowScore,
    _open_shadow_count,
    _record_snapshot,
    _shadow_equity,
    _update_shadow_book,
    compute_shadow_scores,
    snapshot_forward_ab,
)
from engine.signal_aggregator import AggregatedSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = 1_700_000_000


def _make_agg(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    composite_score: float = 0.5,
    action: str = "buy",
    momentum_score: float = 0.3,
    volatility_score: float = 0.2,
    funding_score: float | None = 0.1,
    sentiment_score: float = 0.4,
    ignition_score: float | None = None,
) -> AggregatedSignal:
    return AggregatedSignal(
        symbol=symbol,
        asset_class=asset_class,
        signal_id=1,
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        funding_score=funding_score,
        sentiment_score=sentiment_score,
        composite_score=composite_score,
        action=action,
        ignition_score=ignition_score,
    )


def _make_shadow_score(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    keyless_sentiment: float = 0.3,
    shadow_composite: float = 0.5,
    shadow_action: str = "buy",
) -> ShadowScore:
    return ShadowScore(
        symbol=symbol,
        asset_class=asset_class,
        keyless_sentiment=keyless_sentiment,
        shadow_composite=shadow_composite,
        shadow_action=shadow_action,
    )


def _in_memory_conn() -> sqlite3.Connection:
    """In-memory DB with all tables shadow_tracker uses."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            timeframe   TEXT NOT NULL DEFAULT '1h',
            ts          INTEGER NOT NULL,
            open        REAL NOT NULL DEFAULT 0,
            high        REAL NOT NULL DEFAULT 0,
            low         REAL NOT NULL DEFAULT 0,
            close       REAL NOT NULL,
            volume      REAL NOT NULL DEFAULT 0,
            inserted_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE shadow_positions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol         TEXT NOT NULL,
            asset_class    TEXT NOT NULL,
            side           TEXT NOT NULL DEFAULT 'long',
            quantity       REAL NOT NULL,
            entry_price    REAL NOT NULL,
            current_price  REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            realized_pnl   REAL NOT NULL DEFAULT 0.0,
            status         TEXT NOT NULL DEFAULT 'open',
            opened_at      INTEGER NOT NULL DEFAULT (unixepoch()),
            closed_at      INTEGER
        );
        CREATE TABLE forward_shadow_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           INTEGER NOT NULL DEFAULT (unixepoch()),
            variant      TEXT NOT NULL,
            equity       REAL NOT NULL,
            peak_equity  REAL NOT NULL,
            drawdown_pct REAL NOT NULL DEFAULT 0.0,
            is_paper     INTEGER NOT NULL DEFAULT 1
        );
    """)
    conn.commit()
    return conn


def _insert_price(conn: sqlite3.Connection, symbol: str, close: float) -> None:
    conn.execute(
        "INSERT INTO prices (symbol, ts, close) VALUES (?, ?, ?)",
        (symbol, int(time.time()), close),
    )
    conn.commit()


def _insert_open_shadow_pos(
    conn: sqlite3.Connection,
    symbol: str,
    qty: float = 1.0,
    entry: float = 100.0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO shadow_positions
            (symbol, asset_class, side, quantity, entry_price, current_price)
        VALUES (?, 'crypto', 'long', ?, ?, ?)
        """,
        (symbol, qty, entry, entry),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# compute_shadow_scores
# ---------------------------------------------------------------------------


class TestComputeShadowScores:
    def test_returns_one_score_per_signal(self, monkeypatch):
        from sentiment.sources_crypto import CryptoSources

        def fake_fetch(symbol):
            return CryptoSources(symbol=symbol, fetched_at=0, fear_greed_value=60)

        monkeypatch.setattr("engine.shadow_tracker.sources_crypto.fetch", fake_fetch)
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.pre_score", lambda s: 0.2
        )

        agg = [_make_agg("BTC/USDT"), _make_agg("ETH/USDT", composite_score=0.3)]
        results = compute_shadow_scores(agg)
        assert len(results) == 2
        assert {r.symbol for r in results} == {"BTC/USDT", "ETH/USDT"}

    def test_uses_keyless_pre_score_not_claude(self, monkeypatch):
        from sentiment.sources_crypto import CryptoSources

        called_fetch: list[str] = []

        def fake_fetch(symbol):
            called_fetch.append(symbol)
            return CryptoSources(symbol=symbol, fetched_at=0, fear_greed_value=50)

        monkeypatch.setattr("engine.shadow_tracker.sources_crypto.fetch", fake_fetch)
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.pre_score", lambda s: 0.0
        )

        compute_shadow_scores([_make_agg("BTC/USDT")])
        assert "BTC/USDT" in called_fetch

    def test_skips_symbol_on_fetch_failure(self, monkeypatch):
        from sentiment.sources_crypto import CryptoSources

        def failing_fetch(symbol):
            if symbol == "BTC/USDT":
                raise RuntimeError("network error")
            return CryptoSources(symbol=symbol, fetched_at=0, fear_greed_value=50)

        monkeypatch.setattr("engine.shadow_tracker.sources_crypto.fetch", failing_fetch)
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.pre_score", lambda s: 0.1
        )

        agg = [_make_agg("BTC/USDT"), _make_agg("ETH/USDT")]
        results = compute_shadow_scores(agg)
        assert len(results) == 1
        assert results[0].symbol == "ETH/USDT"

    def test_crypto_uses_sources_crypto(self, monkeypatch):
        from sentiment.sources_crypto import CryptoSources

        crypto_fetched: list[str] = []
        stocks_fetched: list[str] = []

        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.fetch",
            lambda s: (crypto_fetched.append(s), CryptoSources(symbol=s, fetched_at=0))[1],
        )
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.pre_score", lambda s: 0.0
        )
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_stocks.fetch",
            lambda s, **kw: (stocks_fetched.append(s), MagicMock())[1],
        )
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_stocks.pre_score", lambda s: 0.0
        )

        compute_shadow_scores([_make_agg("BTC/USDT", asset_class="crypto")])
        assert "BTC/USDT" in crypto_fetched
        assert "BTC/USDT" not in stocks_fetched

    def test_stocks_passes_empty_finnhub_key(self, monkeypatch):
        received_keys: list[str] = []

        def fake_fetch(symbol, *, finnhub_api_key=""):
            received_keys.append(finnhub_api_key)
            return MagicMock()

        monkeypatch.setattr("engine.shadow_tracker.sources_stocks.fetch", fake_fetch)
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_stocks.pre_score", lambda s: 0.0
        )

        compute_shadow_scores([_make_agg("AAPL", asset_class="stocks")])
        assert received_keys == [""]

    def test_shadow_composite_uses_keyless_sentiment(self, monkeypatch):
        from sentiment.sources_crypto import CryptoSources

        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.fetch",
            lambda s: CryptoSources(symbol=s, fetched_at=0, fear_greed_value=100),
        )
        # Extreme greed → keyless score = +1.0
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.pre_score", lambda s: 1.0
        )

        agg = _make_agg("BTC/USDT", momentum_score=0.0, volatility_score=0.0, funding_score=0.0)
        results = compute_shadow_scores([agg])
        assert len(results) == 1
        assert results[0].keyless_sentiment == pytest.approx(1.0)
        # Composite dominated by +1 sentiment → should be positive
        assert results[0].shadow_composite > 0

    def test_shadow_score_action_in_valid_set(self, monkeypatch):
        from sentiment.sources_crypto import CryptoSources

        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.fetch",
            lambda s: CryptoSources(symbol=s, fetched_at=0),
        )
        monkeypatch.setattr(
            "engine.shadow_tracker.sources_crypto.pre_score", lambda s: 0.0
        )

        results = compute_shadow_scores([_make_agg()])
        assert results[0].shadow_action in {"buy", "sell", "hold"}


# ---------------------------------------------------------------------------
# _update_shadow_book / _shadow_equity
# ---------------------------------------------------------------------------


class TestUpdateShadowBook:
    def _setup(self, monkeypatch):
        conn = _in_memory_conn()
        monkeypatch.setattr("engine.shadow_tracker.get_connection", lambda: conn)
        return conn

    def test_opens_position_on_buy(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 30_000.0)

        _update_shadow_book([_make_shadow_score(shadow_action="buy")])

        row = conn.execute(
            "SELECT * FROM shadow_positions WHERE symbol = 'BTC/USDT' AND status = 'open'"
        ).fetchone()
        assert row is not None
        assert float(row["quantity"]) > 0

    def test_no_duplicate_position_on_buy(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 30_000.0)
        _insert_open_shadow_pos(conn, "BTC/USDT")

        _update_shadow_book([_make_shadow_score(shadow_action="buy")])

        count = conn.execute(
            "SELECT COUNT(*) FROM shadow_positions WHERE symbol = 'BTC/USDT' AND status = 'open'"
        ).fetchone()[0]
        assert count == 1

    def test_closes_position_on_sell(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 35_000.0)
        _insert_open_shadow_pos(conn, "BTC/USDT", qty=0.1, entry=30_000.0)

        _update_shadow_book([_make_shadow_score(shadow_action="sell")])

        closed = conn.execute(
            "SELECT * FROM shadow_positions WHERE symbol = 'BTC/USDT' AND status = 'closed'"
        ).fetchone()
        assert closed is not None
        assert float(closed["realized_pnl"]) > 0  # exited at higher price

    def test_no_action_on_hold(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 30_000.0)

        _update_shadow_book([_make_shadow_score(shadow_action="hold")])

        count = conn.execute(
            "SELECT COUNT(*) FROM shadow_positions"
        ).fetchone()[0]
        assert count == 0

    def test_sell_without_position_is_noop(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 30_000.0)

        _update_shadow_book([_make_shadow_score(shadow_action="sell")])

        count = conn.execute(
            "SELECT COUNT(*) FROM shadow_positions"
        ).fetchone()[0]
        assert count == 0

    def test_respects_max_open_positions(self, monkeypatch):
        import config
        conn = self._setup(monkeypatch)

        # Fill up to MAX_OPEN_POSITIONS with existing positions
        for i in range(config.MAX_OPEN_POSITIONS):
            sym = f"COIN{i}/USDT"
            _insert_price(conn, sym, 100.0)
            _insert_open_shadow_pos(conn, sym)

        # Try to open one more
        _insert_price(conn, "EXTRA/USDT", 100.0)
        _update_shadow_book([_make_shadow_score("EXTRA/USDT", shadow_action="buy")])

        count = conn.execute(
            "SELECT COUNT(*) FROM shadow_positions WHERE status = 'open'"
        ).fetchone()[0]
        assert count == config.MAX_OPEN_POSITIONS

    def test_mark_to_market_updates_unrealized_pnl(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 35_000.0)
        _insert_open_shadow_pos(conn, "BTC/USDT", qty=0.1, entry=30_000.0)

        # 'hold' triggers mark-to-market without opening/closing
        _update_shadow_book([_make_shadow_score(shadow_action="hold")])

        row = conn.execute(
            "SELECT unrealized_pnl, current_price FROM shadow_positions"
            " WHERE symbol = 'BTC/USDT' AND status = 'open'"
        ).fetchone()
        assert row is not None
        # unrealized = (35000 - 30000) * 0.1 = 500
        assert float(row["unrealized_pnl"]) == pytest.approx(500.0)
        assert float(row["current_price"]) == pytest.approx(35_000.0)

    def test_shadow_equity_increases_on_profit(self, monkeypatch):
        import config
        conn = self._setup(monkeypatch)
        _insert_price(conn, "BTC/USDT", 35_000.0)
        _insert_open_shadow_pos(conn, "BTC/USDT", qty=0.1, entry=30_000.0)

        _update_shadow_book([_make_shadow_score(shadow_action="hold")])

        eq = _shadow_equity()
        assert eq > config.INITIAL_CAPITAL


# ---------------------------------------------------------------------------
# _record_snapshot / peak tracking
# ---------------------------------------------------------------------------


class TestRecordSnapshot:
    def _setup(self, monkeypatch):
        conn = _in_memory_conn()
        monkeypatch.setattr("engine.shadow_tracker.get_connection", lambda: conn)
        return conn

    def test_first_snapshot_sets_peak_equal_to_equity(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _record_snapshot("real", 10_500.0)
        row = conn.execute(
            "SELECT peak_equity, drawdown_pct FROM forward_shadow_snapshots WHERE variant = 'real'"
        ).fetchone()
        assert float(row["peak_equity"]) == pytest.approx(10_500.0)
        assert float(row["drawdown_pct"]) == pytest.approx(0.0)

    def test_peak_advances_on_new_high(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _record_snapshot("real", 10_000.0)
        _record_snapshot("real", 11_000.0)
        row = conn.execute(
            "SELECT peak_equity FROM forward_shadow_snapshots"
            " WHERE variant = 'real' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert float(row["peak_equity"]) == pytest.approx(11_000.0)

    def test_drawdown_computed_on_decline(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _record_snapshot("real", 10_000.0)
        _record_snapshot("real", 9_000.0)
        row = conn.execute(
            "SELECT drawdown_pct FROM forward_shadow_snapshots"
            " WHERE variant = 'real' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        # drawdown = (10000 - 9000) / 10000 = 0.1
        assert float(row["drawdown_pct"]) == pytest.approx(0.1)

    def test_real_and_shadow_snapshots_are_independent(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _record_snapshot("real", 10_000.0)
        _record_snapshot("real", 11_000.0)
        _record_snapshot("shadow", 10_500.0)

        real_peak = conn.execute(
            "SELECT peak_equity FROM forward_shadow_snapshots"
            " WHERE variant = 'real' ORDER BY id DESC LIMIT 1"
        ).fetchone()["peak_equity"]
        shadow_peak = conn.execute(
            "SELECT peak_equity FROM forward_shadow_snapshots"
            " WHERE variant = 'shadow' ORDER BY id DESC LIMIT 1"
        ).fetchone()["peak_equity"]

        assert float(real_peak) == pytest.approx(11_000.0)
        assert float(shadow_peak) == pytest.approx(10_500.0)

    def test_is_paper_flag_stored(self, monkeypatch):
        conn = self._setup(monkeypatch)
        _record_snapshot("real", 10_000.0, is_paper=False)
        row = conn.execute(
            "SELECT is_paper FROM forward_shadow_snapshots WHERE variant = 'real'"
        ).fetchone()
        assert row["is_paper"] == 0


# ---------------------------------------------------------------------------
# snapshot_forward_ab — integration
# ---------------------------------------------------------------------------


class TestSnapshotForwardAB:
    def _setup(self, monkeypatch):
        conn = _in_memory_conn()
        monkeypatch.setattr("engine.shadow_tracker.get_connection", lambda: conn)
        return conn

    def _mock_shadow_scores(self, monkeypatch, scores: list[ShadowScore]) -> None:
        monkeypatch.setattr(
            "engine.shadow_tracker.compute_shadow_scores",
            lambda agg: scores,
        )

    def test_persists_real_snapshot(self, monkeypatch):
        conn = self._setup(monkeypatch)
        self._mock_shadow_scores(monkeypatch, [])

        snapshot_forward_ab(10_000.0, [])

        row = conn.execute(
            "SELECT equity FROM forward_shadow_snapshots WHERE variant = 'real'"
        ).fetchone()
        assert row is not None
        assert float(row["equity"]) == pytest.approx(10_000.0)

    def test_persists_shadow_snapshot(self, monkeypatch):
        conn = self._setup(monkeypatch)
        self._mock_shadow_scores(monkeypatch, [])

        snapshot_forward_ab(10_000.0, [])

        row = conn.execute(
            "SELECT equity FROM forward_shadow_snapshots WHERE variant = 'shadow'"
        ).fetchone()
        assert row is not None

    def test_real_snapshot_survives_shadow_failure(self, monkeypatch):
        conn = self._setup(monkeypatch)

        def exploding_scores(_):
            raise RuntimeError("shadow fetch died")

        monkeypatch.setattr(
            "engine.shadow_tracker.compute_shadow_scores", exploding_scores
        )

        snapshot_forward_ab(9_500.0, [])

        real_row = conn.execute(
            "SELECT equity FROM forward_shadow_snapshots WHERE variant = 'real'"
        ).fetchone()
        assert real_row is not None
        assert float(real_row["equity"]) == pytest.approx(9_500.0)

        # Shadow row is absent (computation failed)
        shadow_row = conn.execute(
            "SELECT equity FROM forward_shadow_snapshots WHERE variant = 'shadow'"
        ).fetchone()
        assert shadow_row is None

    def test_is_paper_forwarded_to_both_snapshots(self, monkeypatch):
        conn = self._setup(monkeypatch)
        self._mock_shadow_scores(monkeypatch, [])

        snapshot_forward_ab(10_000.0, [], is_paper=False)

        rows = conn.execute(
            "SELECT variant, is_paper FROM forward_shadow_snapshots"
        ).fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["is_paper"] == 0

    def test_two_snapshots_written_per_cycle(self, monkeypatch):
        conn = self._setup(monkeypatch)
        self._mock_shadow_scores(monkeypatch, [])

        snapshot_forward_ab(10_000.0, [])

        count = conn.execute(
            "SELECT COUNT(*) FROM forward_shadow_snapshots"
        ).fetchone()[0]
        assert count == 2
