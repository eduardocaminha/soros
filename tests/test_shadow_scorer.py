"""Tests for engine.shadow_scorer — forward shadow scoring."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import config
from engine.shadow_scorer import (
    VARIANT_REAL,
    VARIANT_SHADOW,
    ShadowScore,
    _close_position,
    _open_position,
    _record_equity_snapshot,
    _virtual_equity,
    compute_keyless_sentiment,
    compute_shadow_scores,
    tick,
)
from engine.signal_aggregator import AggregatedSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn(tmp_path):
    """Return an in-memory SQLite connection with required tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1h',
            ts INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            funding_rate REAL,
            inserted_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE forward_shadow_positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            variant     TEXT    NOT NULL CHECK (variant IN ('real', 'shadow')),
            symbol      TEXT    NOT NULL,
            asset_class TEXT    NOT NULL,
            side        TEXT    NOT NULL DEFAULT 'long',
            status      TEXT    NOT NULL DEFAULT 'open',
            quantity    REAL    NOT NULL,
            entry_price REAL    NOT NULL,
            exit_price  REAL,
            opened_at   INTEGER NOT NULL DEFAULT (unixepoch()),
            closed_at   INTEGER
        );
        CREATE TABLE forward_shadow_equity (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           INTEGER NOT NULL DEFAULT (unixepoch()),
            variant      TEXT    NOT NULL,
            equity       REAL    NOT NULL,
            peak_equity  REAL    NOT NULL,
            drawdown_pct REAL    NOT NULL DEFAULT 0.0,
            is_paper     INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    return conn


def _make_signal(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    momentum: float = 0.3,
    volatility: float = 0.2,
    funding: float | None = 0.0,
    sentiment: float = 0.0,
    composite: float = 0.25,
    action: str = "buy",
    ignition: float | None = None,
) -> AggregatedSignal:
    return AggregatedSignal(
        symbol=symbol,
        asset_class=asset_class,
        signal_id=1,
        momentum_score=momentum,
        volatility_score=volatility,
        funding_score=funding,
        sentiment_score=sentiment,
        composite_score=composite,
        action=action,
        ignition_score=ignition,
    )


# ---------------------------------------------------------------------------
# compute_keyless_sentiment
# ---------------------------------------------------------------------------

class TestComputeKeylessSentiment:
    def test_stocks_always_zero(self):
        assert compute_keyless_sentiment("AAPL", "stocks") == 0.0

    def test_crypto_with_fg_only(self):
        with (
            patch(
                "engine.shadow_scorer._fetch_fear_greed", return_value=(75, "Greed")
            ),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
        ):
            score = compute_keyless_sentiment("BTC/USDT", "crypto")
        # F&G 75 → (75-50)/50 = 0.5
        assert score == pytest.approx(0.5)

    def test_crypto_with_coingecko_only(self):
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(None, None)),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=0.4
            ),
        ):
            score = compute_keyless_sentiment("BTC/USDT", "crypto")
        assert score == pytest.approx(0.4)

    def test_crypto_averages_both(self):
        with (
            patch(
                "engine.shadow_scorer._fetch_fear_greed", return_value=(50, "Neutral")
            ),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=0.6
            ),
        ):
            score = compute_keyless_sentiment("BTC/USDT", "crypto")
        # F&G 50 → 0.0; CG → 0.6; avg = 0.3
        assert score == pytest.approx(0.3)

    def test_both_none_returns_zero(self):
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(None, None)),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
        ):
            score = compute_keyless_sentiment("BTC/USDT", "crypto")
        assert score == 0.0

    def test_pre_fetched_fg_value_used(self):
        """fg_value kwarg skips the HTTP call."""
        with patch(
            "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
        ):
            score = compute_keyless_sentiment("ETH/USDT", "crypto", fg_value=0)
        # F&G 0 → (0-50)/50 = -1.0
        assert score == pytest.approx(-1.0)

    def test_extreme_fear_clamped(self):
        # F&G 0 → -1.0 (already in bounds), CoinGecko -1.0 → avg -1.0
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(0, "Extreme Fear")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=-1.0
            ),
        ):
            score = compute_keyless_sentiment("BTC/USDT", "crypto")
        assert score == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# compute_shadow_scores
# ---------------------------------------------------------------------------

class TestComputeShadowScores:
    def test_returns_shadow_score_per_signal(self):
        sig = _make_signal(action="buy", composite=0.3)
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(60, "Greed")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=0.2
            ),
            patch("engine.shadow_scorer.get_connection") as mock_conn,
        ):
            mock_conn.return_value.execute.return_value.fetchone.return_value = {"close": 50_000.0}
            results = compute_shadow_scores([sig])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, ShadowScore)
        assert r.symbol == "BTC/USDT"
        assert r.real_composite == pytest.approx(0.3)
        assert r.real_action == "buy"

    def test_shadow_composite_differs_from_real_when_sentiment_differs(self):
        """When keyless sentiment differs from real sentiment the composites differ."""
        sig = _make_signal(
            momentum=0.0,
            volatility=0.0,
            funding=0.0,
            sentiment=0.8,   # real used high sentiment
            composite=0.4,
            action="buy",
        )
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(50, "Neutral")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
            patch("engine.shadow_scorer.get_connection") as mock_conn,
        ):
            mock_conn.return_value.execute.return_value.fetchone.return_value = None
            results = compute_shadow_scores([sig])
        # keyless sentiment = 0.0; real used 0.8 → shadow composite should differ
        assert len(results) == 1
        r = results[0]
        # shadow_composite uses keyless sent=0.0; real_composite=0.4 (from sig)
        assert r.real_composite == pytest.approx(0.4)
        # shadow composite is independently computed with sent=0.0
        assert r.shadow_composite != pytest.approx(r.real_composite, rel=1e-2)

    def test_current_price_none_when_no_price_row(self):
        sig = _make_signal()
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(None, None)),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
            patch("engine.shadow_scorer.get_connection") as mock_conn,
        ):
            mock_conn.return_value.execute.return_value.fetchone.return_value = None
            results = compute_shadow_scores([sig])
        assert results[0].current_price is None

    def test_stocks_symbol_gets_zero_keyless_sentiment(self):
        sig = _make_signal(symbol="AAPL", asset_class="stocks", funding=None)
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(70, "Greed")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=0.5
            ),
            patch("engine.shadow_scorer.get_connection") as mock_conn,
        ):
            mock_conn.return_value.execute.return_value.fetchone.return_value = None
            results = compute_shadow_scores([sig])
        assert len(results) == 1
        assert results[0].keyless_sentiment == 0.0

    def test_fg_fetched_once_for_multiple_crypto_symbols(self):
        signals = [
            _make_signal("BTC/USDT", "crypto"),
            _make_signal("ETH/USDT", "crypto"),
        ]
        with (
            patch(
                "engine.shadow_scorer._fetch_fear_greed",
                return_value=(55, "Greed"),
            ) as mock_fg,
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
            patch("engine.shadow_scorer.get_connection") as mock_conn,
        ):
            mock_conn.return_value.execute.return_value.fetchone.return_value = None
            compute_shadow_scores(signals)
        # F&G fetched once at the top, not once per symbol
        mock_fg.assert_called_once()

    def test_failed_symbol_is_skipped_not_raised(self):
        sig = _make_signal()
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", side_effect=Exception("net")),
            patch("engine.shadow_scorer.get_connection") as mock_conn,
        ):
            mock_conn.return_value.execute.return_value.fetchone.return_value = None
            results = compute_shadow_scores([sig])
        assert results == []


# ---------------------------------------------------------------------------
# Virtual position helpers (using patched get_connection)
# ---------------------------------------------------------------------------

class TestVirtualEquity:
    def test_initial_equity_equals_initial_capital(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            conn.execute.return_value.fetchone.return_value = (0.0,)
            conn.execute.return_value.fetchall.return_value = []
            eq = _virtual_equity(VARIANT_REAL)
        assert eq == pytest.approx(config.INITIAL_CAPITAL)

    def test_open_position_adds_unrealized(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            # Simulate: realized=0, 1 open position qty=1, entry=100, current=110
            call_count = [0]

            def _exe(sql, params=()):
                call_count[0] += 1
                m = MagicMock()
                if "SUM((exit_price" in sql:
                    m.fetchone.return_value = (0.0,)
                elif "from forward_shadow_positions" in sql.lower() and "open" in sql:
                    m.fetchall.return_value = [{"symbol": "BTC/USDT", "quantity": 1.0, "entry_price": 100.0}]
                elif "from prices" in sql.lower():
                    m.fetchone.return_value = {"close": 110.0}
                else:
                    m.fetchone.return_value = None
                    m.fetchall.return_value = []
                return m

            conn.execute.side_effect = _exe
            eq = _virtual_equity(VARIANT_REAL)
        # unrealized = (110 - 100) * 1 = 10
        assert eq == pytest.approx(config.INITIAL_CAPITAL + 10.0)


class TestOpenClosePosition:
    def test_open_inserts_row_when_no_existing(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            conn.execute.return_value.fetchone.return_value = None  # no existing
            _open_position(VARIANT_SHADOW, "BTC/USDT", "crypto", 50_000.0, 10_000.0)
            # verify INSERT was called
            calls = [str(c) for c in conn.execute.call_args_list]
            assert any("INSERT" in c for c in calls)

    def test_open_skips_when_already_open(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            existing = MagicMock()
            existing.__getitem__ = lambda s, k: 1
            conn.execute.return_value.fetchone.return_value = existing
            _open_position(VARIANT_SHADOW, "BTC/USDT", "crypto", 50_000.0, 10_000.0)
            # INSERT should NOT be called
            calls = [str(c) for c in conn.execute.call_args_list]
            assert not any("INSERT" in c for c in calls)

    def test_close_updates_row_when_open(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            # sqlite3.Row-like: support ["id"] access via plain dict
            conn.execute.return_value.fetchone.return_value = {"id": 42}
            _close_position(VARIANT_REAL, "BTC/USDT", 52_000.0)
            calls = [str(c) for c in conn.execute.call_args_list]
            assert any("UPDATE" in c for c in calls)

    def test_close_noop_when_no_open_position(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            conn.execute.return_value.fetchone.return_value = None
            _close_position(VARIANT_REAL, "BTC/USDT", 52_000.0)
            calls = [str(c) for c in conn.execute.call_args_list]
            assert not any("UPDATE" in c for c in calls)


# ---------------------------------------------------------------------------
# _record_equity_snapshot
# ---------------------------------------------------------------------------

class TestRecordEquitySnapshot:
    def test_inserts_first_snapshot(self):
        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            conn = mock_conn.return_value
            conn.execute.return_value.fetchone.return_value = None  # no prior row
            _record_equity_snapshot(VARIANT_SHADOW, 10_500.0, True)
            calls = [str(c) for c in conn.execute.call_args_list]
            assert any("INSERT" in c for c in calls)

    def test_peak_advances_on_new_high(self):
        captured: list[tuple] = []

        def _exe(sql, params=()):
            m = MagicMock()
            if "peak_equity" in sql and "SELECT" in sql:
                m.fetchone.return_value = {"peak_equity": 10_000.0}
            else:
                if "INSERT" in sql:
                    captured.append(params)
            return m

        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            mock_conn.return_value.execute.side_effect = _exe
            _record_equity_snapshot(VARIANT_REAL, 11_000.0, True)

        assert len(captured) == 1
        _variant, equity, peak, drawdown, _is_paper = captured[0]
        assert equity == pytest.approx(11_000.0)
        assert peak == pytest.approx(11_000.0)
        assert drawdown == pytest.approx(0.0)

    def test_drawdown_computed_when_below_peak(self):
        captured: list[tuple] = []

        def _exe(sql, params=()):
            m = MagicMock()
            if "peak_equity" in sql and "SELECT" in sql:
                m.fetchone.return_value = {"peak_equity": 10_000.0}
            else:
                if "INSERT" in sql:
                    captured.append(params)
            return m

        with patch("engine.shadow_scorer.get_connection") as mock_conn:
            mock_conn.return_value.execute.side_effect = _exe
            _record_equity_snapshot(VARIANT_SHADOW, 9_000.0, True)

        assert len(captured) == 1
        _variant, equity, peak, drawdown, _is_paper = captured[0]
        assert equity == pytest.approx(9_000.0)
        assert peak == pytest.approx(10_000.0)
        # drawdown = (10000 - 9000) / 10000 = 0.1
        assert drawdown == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# tick — main entry point
# ---------------------------------------------------------------------------

class TestTick:
    def _patch_tick(self):
        """Context managers to isolate tick() from real I/O."""
        return (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(55, "Greed")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=0.1
            ),
            patch("engine.shadow_scorer._virtual_equity", return_value=10_000.0),
            patch("engine.shadow_scorer._open_position"),
            patch("engine.shadow_scorer._close_position"),
            patch("engine.shadow_scorer._record_equity_snapshot"),
            patch("engine.shadow_scorer.get_connection"),
        )

    def test_returns_list_of_shadow_scores(self):
        sig = _make_signal(action="buy")
        patches = self._patch_tick()

        def _exe(sql, params=()):
            m = MagicMock()
            m.fetchone.return_value = {"close": 50_000.0}
            m.fetchall.return_value = []
            return m

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6] as mc:
            mc.return_value.execute.side_effect = _exe
            results = tick([sig])

        assert len(results) == 1
        assert isinstance(results[0], ShadowScore)

    def test_empty_aggregated_returns_empty(self):
        results = tick([])
        assert results == []

    def test_shadow_failure_propagates_to_caller(self):
        """tick() propagates exceptions; the caller (main.py) wraps in try/except."""
        sig = _make_signal()
        with patch(
            "engine.shadow_scorer.compute_shadow_scores",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                tick([sig])

    def test_symbol_with_no_price_is_skipped_gracefully(self):
        sig = _make_signal(action="buy")
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(55, "Greed")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
            patch("engine.shadow_scorer._open_position") as mock_open,
            patch("engine.shadow_scorer._close_position") as mock_close,
            patch("engine.shadow_scorer._virtual_equity", return_value=10_000.0),
            patch("engine.shadow_scorer._record_equity_snapshot"),
            patch("engine.shadow_scorer.get_connection") as mc,
        ):
            mc.return_value.execute.return_value.fetchone.return_value = None
            results = tick([sig])

        # price was None → position update skipped
        mock_open.assert_not_called()
        mock_close.assert_not_called()
        assert len(results) == 1

    def test_buy_action_triggers_open_for_both_variants(self):
        sig = _make_signal(action="buy", composite=0.5)
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(80, "Extreme Greed")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=0.8
            ),
            patch("engine.shadow_scorer._open_position") as mock_open,
            patch("engine.shadow_scorer._close_position"),
            patch("engine.shadow_scorer._virtual_equity", return_value=10_000.0),
            patch("engine.shadow_scorer._record_equity_snapshot"),
            patch("engine.shadow_scorer.get_connection") as mc,
        ):
            mc.return_value.execute.return_value.fetchone.return_value = {"close": 50_000.0}
            tick([sig])

        # open_position must be called for each variant that has action=buy
        variants_opened = {c.args[0] for c in mock_open.call_args_list}
        assert VARIANT_REAL in variants_opened

    def test_sell_action_triggers_close(self):
        sig = _make_signal(action="sell", composite=-0.6)
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(20, "Fear")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=-0.5
            ),
            patch("engine.shadow_scorer._open_position"),
            patch("engine.shadow_scorer._close_position") as mock_close,
            patch("engine.shadow_scorer._virtual_equity", return_value=9_500.0),
            patch("engine.shadow_scorer._record_equity_snapshot"),
            patch("engine.shadow_scorer.get_connection") as mc,
        ):
            mc.return_value.execute.return_value.fetchone.return_value = {"close": 48_000.0}
            tick([sig])

        variants_closed = {c.args[0] for c in mock_close.call_args_list}
        # real action=sell → close for real variant
        assert VARIANT_REAL in variants_closed

    def test_equity_snapshot_recorded_for_both_variants(self):
        sig = _make_signal(action="hold")
        with (
            patch("engine.shadow_scorer._fetch_fear_greed", return_value=(50, "Neutral")),
            patch(
                "engine.shadow_scorer._fetch_coingecko_sentiment", return_value=None
            ),
            patch("engine.shadow_scorer._open_position"),
            patch("engine.shadow_scorer._close_position"),
            patch("engine.shadow_scorer._virtual_equity", return_value=10_000.0),
            patch("engine.shadow_scorer._record_equity_snapshot") as mock_snap,
            patch("engine.shadow_scorer.get_connection") as mc,
        ):
            mc.return_value.execute.return_value.fetchone.return_value = None
            tick([sig])

        snapped_variants = {c.args[0] for c in mock_snap.call_args_list}
        assert VARIANT_REAL in snapped_variants
        assert VARIANT_SHADOW in snapped_variants
