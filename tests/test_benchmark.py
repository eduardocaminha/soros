"""Tests for engine/benchmark.py — pure benchmark computation + DB helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from engine.benchmark import (
    BenchmarkSeries,
    build_btc_benchmark,
    load_btc_closes,
    load_equity_snapshots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snaps(*pairs: tuple[int, float]) -> list[tuple[int, float]]:
    return list(pairs)


def _btc(*pairs: tuple[int, float]) -> list[tuple[int, float]]:
    return list(pairs)


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    schema = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    conn.executescript(schema)
    conn.commit()
    return conn


def _insert_equity(conn: sqlite3.Connection, ts: int, equity: float, is_paper: bool = True) -> None:
    conn.execute(
        "INSERT INTO equity_curve (ts, equity, peak_equity, drawdown_pct, is_paper)"
        " VALUES (?, ?, ?, 0.0, ?)",
        (ts, equity, equity, int(is_paper)),
    )
    conn.commit()


def _insert_price(conn: sqlite3.Connection, ts: int, close: float, symbol: str = "BTC/USDT") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO prices"
        " (symbol, asset_class, timeframe, ts, open, high, low, close, volume)"
        " VALUES (?, 'crypto', '1h', ?, ?, ?, ?, ?, 1.0)",
        (symbol, ts, close, close, close, close),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# build_btc_benchmark — basic correctness
# ---------------------------------------------------------------------------

class TestBuildBtcBenchmarkBasic:
    def test_single_point_series(self):
        result = build_btc_benchmark(
            snapshots=[(1000, 10_000.0)],
            btc_closes=[(1000, 50_000.0)],
        )
        assert result.n_points == 1
        assert result.initial_capital == pytest.approx(10_000.0)
        assert result.btc_start_price == pytest.approx(50_000.0)
        assert result.btc_equity[0] == pytest.approx(10_000.0)  # no change

    def test_btc_doubled_doubles_equity(self):
        result = build_btc_benchmark(
            snapshots=[(1000, 10_000.0), (2000, 9_000.0)],
            btc_closes=[(1000, 50_000.0), (2000, 100_000.0)],
        )
        assert result.btc_equity[0] == pytest.approx(10_000.0)
        assert result.btc_equity[1] == pytest.approx(20_000.0)

    def test_btc_halved_halves_equity(self):
        result = build_btc_benchmark(
            snapshots=[(1000, 10_000.0), (2000, 12_000.0)],
            btc_closes=[(1000, 40_000.0), (2000, 20_000.0)],
        )
        assert result.btc_equity[1] == pytest.approx(5_000.0)

    def test_initial_capital_from_first_aligned_snapshot(self):
        result = build_btc_benchmark(
            snapshots=[(1000, 5_000.0), (2000, 6_000.0)],
            btc_closes=[(1000, 30_000.0), (2000, 30_000.0)],
        )
        assert result.initial_capital == pytest.approx(5_000.0)

    def test_soros_equity_matches_input(self):
        snaps = [(1000, 10_000.0), (2000, 11_000.0), (3000, 9_500.0)]
        result = build_btc_benchmark(snaps, [(1000, 50_000.0), (2000, 50_000.0), (3000, 50_000.0)])
        assert result.soros_equity == (10_000.0, 11_000.0, 9_500.0)

    def test_btc_flat_price_equals_initial_capital(self):
        result = build_btc_benchmark(
            snapshots=[(1000, 10_000.0), (2000, 11_000.0), (3000, 12_000.0)],
            btc_closes=[(1000, 50_000.0), (2000, 50_000.0), (3000, 50_000.0)],
        )
        for btc_eq in result.btc_equity:
            assert btc_eq == pytest.approx(10_000.0)

    def test_window_start_and_end(self):
        result = build_btc_benchmark(
            snapshots=[(100, 1_000.0), (200, 1_100.0), (300, 1_050.0)],
            btc_closes=[(100, 50_000.0), (200, 51_000.0), (300, 52_000.0)],
        )
        assert result.window_start == 100
        assert result.window_end == 300

    def test_n_points_matches_output_length(self):
        result = build_btc_benchmark(
            snapshots=[(i * 100, float(1000 + i)) for i in range(5)],
            btc_closes=[(i * 100, float(50_000 + i * 100)) for i in range(5)],
        )
        assert result.n_points == 5
        assert len(result.timestamps) == result.n_points
        assert len(result.soros_equity) == result.n_points
        assert len(result.btc_equity) == result.n_points


# ---------------------------------------------------------------------------
# build_btc_benchmark — unsorted inputs
# ---------------------------------------------------------------------------

class TestBuildBtcBenchmarkUnsorted:
    def test_snapshots_unsorted(self):
        snaps = [(300, 12_000.0), (100, 10_000.0), (200, 11_000.0)]
        btc = [(100, 50_000.0), (200, 51_000.0), (300, 52_000.0)]
        result = build_btc_benchmark(snaps, btc)
        # After sorting, first snapshot is ts=100 with equity=10_000
        assert result.initial_capital == pytest.approx(10_000.0)
        assert result.window_start == 100

    def test_btc_closes_unsorted(self):
        snaps = [(100, 10_000.0), (200, 11_000.0)]
        btc = [(200, 51_000.0), (100, 50_000.0)]
        result = build_btc_benchmark(snaps, btc)
        assert result.btc_start_price == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# build_btc_benchmark — gap handling (forward fill)
# ---------------------------------------------------------------------------

class TestBuildBtcBenchmarkGaps:
    def test_missing_btc_candle_forward_filled(self):
        # ts=200 has no BTC close; should use ts=100 price
        result = build_btc_benchmark(
            snapshots=[(100, 10_000.0), (200, 10_500.0), (300, 11_000.0)],
            btc_closes=[(100, 50_000.0), (300, 60_000.0)],
        )
        assert result.n_points == 3
        assert result.n_btc_gaps == 1
        # ts=200 forward-fills from ts=100 → btc_equity[1] = 10_000 * (50_000/50_000)
        assert result.btc_equity[1] == pytest.approx(10_000.0)

    def test_no_gaps_when_all_prices_present(self):
        result = build_btc_benchmark(
            snapshots=[(100, 10_000.0), (200, 11_000.0)],
            btc_closes=[(100, 50_000.0), (200, 55_000.0)],
        )
        assert result.n_btc_gaps == 0

    def test_multiple_gaps_counted(self):
        result = build_btc_benchmark(
            snapshots=[(100, 10_000.0), (200, 10_200.0), (300, 10_400.0), (400, 10_600.0)],
            btc_closes=[(100, 50_000.0), (400, 60_000.0)],
        )
        # ts=200 and ts=300 are forward-filled from ts=100
        assert result.n_btc_gaps == 2
        assert result.n_points == 4

    def test_btc_price_before_snapshot_window_used(self):
        # BTC has a price at ts=50, before first snapshot at ts=100
        result = build_btc_benchmark(
            snapshots=[(100, 10_000.0), (200, 11_000.0)],
            btc_closes=[(50, 45_000.0), (200, 55_000.0)],
        )
        # First snapshot (ts=100) uses forward fill from ts=50 → btc_start_price = 45_000
        assert result.btc_start_price == pytest.approx(45_000.0)
        assert result.n_btc_gaps == 1  # ts=100 has no exact match
        assert result.btc_equity[0] == pytest.approx(10_000.0)
        # ts=200: 10_000 * (55_000 / 45_000)
        assert result.btc_equity[1] == pytest.approx(10_000.0 * 55_000.0 / 45_000.0)


# ---------------------------------------------------------------------------
# build_btc_benchmark — skipping snapshots before BTC data
# ---------------------------------------------------------------------------

class TestBuildBtcBenchmarkWindowAlignment:
    def test_snapshots_before_btc_data_skipped(self):
        # Snapshots at ts=100,200 but BTC only from ts=200
        result = build_btc_benchmark(
            snapshots=[(100, 10_000.0), (200, 10_500.0), (300, 11_000.0)],
            btc_closes=[(200, 50_000.0), (300, 55_000.0)],
        )
        # ts=100 has no BTC coverage — skipped
        assert result.n_points == 2
        assert result.window_start == 200
        assert result.initial_capital == pytest.approx(10_500.0)

    def test_common_window_declared_in_window_start(self):
        result = build_btc_benchmark(
            snapshots=[(50, 9_000.0), (100, 10_000.0), (150, 11_000.0)],
            btc_closes=[(100, 50_000.0), (150, 52_000.0)],
        )
        assert result.window_start == 100
        assert result.window_end == 150


# ---------------------------------------------------------------------------
# build_btc_benchmark — error cases
# ---------------------------------------------------------------------------

class TestBuildBtcBenchmarkErrors:
    def test_empty_snapshots_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_btc_benchmark([], [(100, 50_000.0)])

    def test_empty_btc_closes_raises(self):
        with pytest.raises(ValueError, match="no BTC close"):
            build_btc_benchmark([(100, 10_000.0)], [])

    def test_btc_all_after_snapshots_raises(self):
        # All BTC prices are after the last snapshot
        with pytest.raises(ValueError, match="no BTC close"):
            build_btc_benchmark(
                [(100, 10_000.0), (200, 11_000.0)],
                [(500, 50_000.0), (600, 55_000.0)],
            )


# ---------------------------------------------------------------------------
# build_btc_benchmark — result is frozen / hashable fields
# ---------------------------------------------------------------------------

class TestBuildBtcBenchmarkImmutability:
    def test_result_is_frozen(self):
        result = build_btc_benchmark(
            [(100, 10_000.0), (200, 11_000.0)],
            [(100, 50_000.0), (200, 55_000.0)],
        )
        with pytest.raises((AttributeError, TypeError)):
            result.n_points = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DB helpers — load_equity_snapshots
# ---------------------------------------------------------------------------

class TestLoadEquitySnapshots:
    def test_empty_table_returns_empty(self, db: sqlite3.Connection):
        assert load_equity_snapshots(db) == []

    def test_rows_returned_ascending(self, db: sqlite3.Connection):
        _insert_equity(db, 300, 12_000.0)
        _insert_equity(db, 100, 10_000.0)
        _insert_equity(db, 200, 11_000.0)
        result = load_equity_snapshots(db)
        assert [ts for ts, _ in result] == [100, 200, 300]

    def test_equity_values_correct(self, db: sqlite3.Connection):
        _insert_equity(db, 100, 10_000.0)
        _insert_equity(db, 200, 11_500.0)
        result = load_equity_snapshots(db)
        assert result[0][1] == pytest.approx(10_000.0)
        assert result[1][1] == pytest.approx(11_500.0)

    def test_filter_by_is_paper_true(self, db: sqlite3.Connection):
        _insert_equity(db, 100, 10_000.0, is_paper=True)
        _insert_equity(db, 200, 9_000.0, is_paper=False)
        result = load_equity_snapshots(db, is_paper=True)
        assert len(result) == 1
        assert result[0][0] == 100

    def test_filter_by_is_paper_false(self, db: sqlite3.Connection):
        _insert_equity(db, 100, 10_000.0, is_paper=True)
        _insert_equity(db, 200, 9_000.0, is_paper=False)
        result = load_equity_snapshots(db, is_paper=False)
        assert len(result) == 1
        assert result[0][0] == 200

    def test_no_filter_returns_all(self, db: sqlite3.Connection):
        _insert_equity(db, 100, 10_000.0, is_paper=True)
        _insert_equity(db, 200, 9_000.0, is_paper=False)
        result = load_equity_snapshots(db)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# DB helpers — load_btc_closes
# ---------------------------------------------------------------------------

class TestLoadBtcCloses:
    def test_empty_table_returns_empty(self, db: sqlite3.Connection):
        assert load_btc_closes(db) == []

    def test_rows_returned_ascending(self, db: sqlite3.Connection):
        _insert_price(db, 300, 52_000.0)
        _insert_price(db, 100, 50_000.0)
        _insert_price(db, 200, 51_000.0)
        result = load_btc_closes(db)
        assert [ts for ts, _ in result] == [100, 200, 300]

    def test_close_values_correct(self, db: sqlite3.Connection):
        _insert_price(db, 100, 48_000.0)
        result = load_btc_closes(db)
        assert result[0][1] == pytest.approx(48_000.0)

    def test_since_filter(self, db: sqlite3.Connection):
        _insert_price(db, 100, 50_000.0)
        _insert_price(db, 200, 51_000.0)
        _insert_price(db, 300, 52_000.0)
        result = load_btc_closes(db, since=200)
        assert [ts for ts, _ in result] == [200, 300]

    def test_until_filter(self, db: sqlite3.Connection):
        _insert_price(db, 100, 50_000.0)
        _insert_price(db, 200, 51_000.0)
        _insert_price(db, 300, 52_000.0)
        result = load_btc_closes(db, until=200)
        assert [ts for ts, _ in result] == [100, 200]

    def test_since_and_until_filter(self, db: sqlite3.Connection):
        for ts in [100, 200, 300, 400]:
            _insert_price(db, ts, float(50_000 + ts))
        result = load_btc_closes(db, since=200, until=300)
        assert [ts for ts, _ in result] == [200, 300]

    def test_other_symbol_excluded(self, db: sqlite3.Connection):
        _insert_price(db, 100, 50_000.0, symbol="BTC/USDT")
        _insert_price(db, 100, 3_000.0, symbol="ETH/USDT")
        result = load_btc_closes(db)
        assert len(result) == 1
        assert result[0][1] == pytest.approx(50_000.0)

    def test_custom_symbol(self, db: sqlite3.Connection):
        _insert_price(db, 100, 3_000.0, symbol="ETH/USDT")
        result = load_btc_closes(db, symbol="ETH/USDT")
        assert len(result) == 1
        assert result[0][1] == pytest.approx(3_000.0)


# ---------------------------------------------------------------------------
# Integration: load + build
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_load_and_build_roundtrip(self, db: sqlite3.Connection):
        _insert_equity(db, 1000, 10_000.0)
        _insert_equity(db, 2000, 11_000.0)
        _insert_equity(db, 3000, 10_500.0)
        _insert_price(db, 1000, 50_000.0)
        _insert_price(db, 2000, 55_000.0)
        _insert_price(db, 3000, 52_500.0)

        snaps = load_equity_snapshots(db)
        closes = load_btc_closes(db)
        result = build_btc_benchmark(snaps, closes)

        assert result.n_points == 3
        assert result.initial_capital == pytest.approx(10_000.0)
        assert result.btc_start_price == pytest.approx(50_000.0)
        # BTC +10%: btc_equity[1] = 10_000 * (55_000/50_000) = 11_000
        assert result.btc_equity[1] == pytest.approx(11_000.0)
        # BTC +5%: btc_equity[2] = 10_000 * (52_500/50_000) = 10_500
        assert result.btc_equity[2] == pytest.approx(10_500.0)

    def test_paper_filter_integration(self, db: sqlite3.Connection):
        _insert_equity(db, 1000, 10_000.0, is_paper=True)
        _insert_equity(db, 2000, 5_000.0, is_paper=False)
        _insert_price(db, 1000, 50_000.0)
        _insert_price(db, 2000, 55_000.0)

        snaps = load_equity_snapshots(db, is_paper=True)
        closes = load_btc_closes(db)
        result = build_btc_benchmark(snaps, closes)

        assert result.n_points == 1
        assert result.initial_capital == pytest.approx(10_000.0)
