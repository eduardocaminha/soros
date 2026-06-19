"""Tests for deterministic signal modules: momentum, volatility, funding, ignition, compute."""

from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from signals import funding, ignition, momentum, volatility
from signals.compute import SignalResult, _action, _deterministic_composite, compute_signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_returns_zero_when_not_enough_data(self):
        s = _series([100.0] * 25)  # _SLOW = 26, need at least 26
        assert momentum.compute(s) == 0.0

    def test_flat_prices_score_near_zero(self):
        s = _series([100.0] * 50)
        assert abs(momentum.compute(s)) < 0.01

    def test_rising_trend_positive(self):
        prices = [100.0 + i for i in range(50)]
        score = momentum.compute(_series(prices))
        assert score > 0.05

    def test_falling_trend_negative(self):
        prices = [200.0 - i for i in range(50)]
        score = momentum.compute(_series(prices))
        assert score < -0.05

    def test_output_bounded(self):
        # Very aggressive trend
        prices = [1.0 * (2 ** i) for i in range(50)]
        score = momentum.compute(_series(prices))
        assert -1.0 <= score <= 1.0

    def test_symmetry(self):
        up = [100.0 + i * 2 for i in range(50)]
        down = [200.0 - i * 2 for i in range(50)]
        assert momentum.compute(_series(up)) > 0
        assert momentum.compute(_series(down)) < 0


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------

def _ohlc_df(closes: list[float], hl_spread: float = 1.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a fixed high/low spread."""
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({"high": c + hl_spread, "low": c - hl_spread, "close": c})


class TestVolatility:
    def test_returns_zero_when_not_enough_data(self):
        # needs _WINDOW + 1 = 21 rows; 20 is insufficient
        df = _ohlc_df([100.0] * 20)
        assert volatility.compute(df) == 0.0

    def test_zero_atr_returns_zero(self):
        # identical OHLC (doji candles) → TR = 0 → ATR = 0 → guard returns 0.0
        df = _ohlc_df([100.0] * 30, hl_spread=0.0)
        assert volatility.compute(df) == 0.0

    def test_price_above_upper_band_clamped_to_one(self):
        # 29 rows at 100, last close = 200 → well above EMA + 2*ATR
        df = _ohlc_df([100.0] * 29 + [200.0])
        assert volatility.compute(df) == 1.0

    def test_price_below_lower_band_clamped_to_neg_one(self):
        # 29 rows at 100, last close = 0 → well below EMA - 2*ATR
        df = _ohlc_df([100.0] * 29 + [0.0])
        assert volatility.compute(df) == -1.0

    def test_price_at_ema_returns_near_zero(self):
        # Steady prices → EMA ≈ mean; close at the mean → score ≈ 0
        prices = [100.0] * 30
        df = _ohlc_df(prices)
        # Close is at the EMA (centre of channel) → linear map gives 0
        score = volatility.compute(df)
        assert abs(score) < 0.1

    def test_output_clamped(self):
        df = _ohlc_df([100.0] * 29 + [1e6])
        assert volatility.compute(df) == 1.0


# ---------------------------------------------------------------------------
# funding
# ---------------------------------------------------------------------------

class TestFunding:
    def test_none_returns_zero(self):
        assert funding.compute(None) == 0.0

    def test_nan_returns_zero(self):
        assert funding.compute(float("nan")) == 0.0

    def test_zero_funding_returns_zero(self):
        assert funding.compute(0.0) == 0.0

    def test_positive_funding_is_bearish(self):
        # Positive funding (longs pay shorts) → contrarian bearish → negative score
        assert funding.compute(0.001) < 0

    def test_negative_funding_is_bullish(self):
        # Negative funding → contrarian bullish → positive score
        assert funding.compute(-0.001) > 0

    def test_symmetry(self):
        pos = funding.compute(0.001)
        neg = funding.compute(-0.001)
        assert abs(pos + neg) < 1e-9

    def test_output_bounded(self):
        assert -1.0 <= funding.compute(1.0) <= 1.0
        assert -1.0 <= funding.compute(-1.0) <= 1.0


# ---------------------------------------------------------------------------
# ignition
# ---------------------------------------------------------------------------

def _ign_df(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": pd.Series(closes, dtype=float), "volume": pd.Series(volumes, dtype=float)})


class TestIgnition:
    def test_returns_zero_when_not_enough_data(self):
        # needs _VOL_WINDOW + 1 = 21 rows minimum
        df = _ign_df([100.0] * 20, [1000.0] * 20)
        assert ignition.compute(df) == 0.0

    def test_flat_prices_flat_volume_near_zero(self):
        # constant volume (std=0) → vol_sig=0; flat close → roc=0 → roc_sig=0
        df = _ign_df([100.0] * 30, [1000.0] * 30)
        assert ignition.compute(df) == 0.0

    def test_volume_surge_positive_price_positive(self):
        # Reference window has varying volume (std > 0); last bar is a 5x spike
        # with rising price → both components positive → clear positive ignition
        ref_vols = [800.0 + (i % 5) * 100 for i in range(29)]  # 800-1200, std ≈ 158
        closes = [100.0 + i * 0.5 for i in range(30)]
        volumes = ref_vols + [5000.0]  # 5x spike well above mean
        df = _ign_df(closes, volumes)
        assert ignition.compute(df) > 0.1

    def test_volume_crash_negative(self):
        # Reference window has varying volume (std > 0); last bar is a tiny crash
        ref_vols = [800.0 + (i % 5) * 100 for i in range(29)]  # 800-1200
        closes = [100.0] * 30  # flat price → roc_sig = 0
        volumes = ref_vols + [10.0]  # deep below mean → z very negative
        df = _ign_df(closes, volumes)
        assert ignition.compute(df) < 0.0

    def test_price_roc_drives_signal_when_volume_flat(self):
        # price jumps significantly, volume flat (std=0) → only roc contributes
        closes = [100.0] * 29 + [115.0]  # +15 % jump
        volumes = [1000.0] * 30  # flat volume → std=0 → vol_sig=0
        df = _ign_df(closes, volumes)
        score = ignition.compute(df)
        # roc = 0.15/5bars_ref which is _ROC_WINDOW=5; ref is closes[-6]=100.0
        # roc_sig = tanh(0.15/0.05) = tanh(3) ≈ 0.995; total = 0.995/2 ≈ 0.497
        assert score > 0.4

    def test_output_bounded(self):
        closes = [1.0 * (2 ** i) for i in range(30)]
        volumes = [1000.0 * (2 ** i) for i in range(30)]
        df = _ign_df(closes, volumes)
        score = ignition.compute(df)
        assert -1.0 <= score <= 1.0

    def test_symmetry_volume_component(self):
        # Reference window has variance; spike should score higher than crash
        ref_vols = [800.0 + (i % 5) * 100 for i in range(29)]  # varying volume
        closes = [100.0] * 30  # flat price (roc = 0) so only volume drives the signal
        spike_df = _ign_df(closes, ref_vols + [5000.0])
        crash_df = _ign_df(closes, ref_vols + [10.0])
        assert ignition.compute(spike_df) > ignition.compute(crash_df)


# ---------------------------------------------------------------------------
# _action
# ---------------------------------------------------------------------------

class TestAction:
    def test_above_threshold_buy(self):
        import config
        assert _action(config.SIGNAL_THRESHOLD + 0.01) == "buy"

    def test_below_neg_threshold_sell(self):
        import config
        assert _action(-(config.SIGNAL_THRESHOLD + 0.01)) == "sell"

    def test_near_zero_hold(self):
        assert _action(0.0) == "hold"


# ---------------------------------------------------------------------------
# _deterministic_composite
# ---------------------------------------------------------------------------

class TestDeterministicComposite:
    def test_crypto_all_positive_returns_positive(self):
        score = _deterministic_composite(0.5, 0.5, 0.5, "crypto")
        assert score > 0

    def test_crypto_all_negative_returns_negative(self):
        score = _deterministic_composite(-0.5, -0.5, -0.5, "crypto")
        assert score < 0

    def test_stocks_ignores_funding_none(self):
        # stocks don't pass funding
        score = _deterministic_composite(0.5, 0.5, None, "stocks")
        assert score > 0

    def test_output_bounded(self):
        score = _deterministic_composite(1.0, 1.0, 1.0, "crypto")
        assert -1.0 <= score <= 1.0

    def test_ignition_increases_positive_composite(self):
        without = _deterministic_composite(0.5, 0.5, 0.5, "crypto")
        with_ign = _deterministic_composite(0.5, 0.5, 0.5, "crypto", ign=1.0)
        assert with_ign > without

    def test_ignition_not_applied_to_stocks(self):
        # stocks never receive ignition; ign param is simply not passed
        score = _deterministic_composite(0.5, 0.5, None, "stocks")
        assert -1.0 <= score <= 1.0

    def test_ignition_none_equals_no_ignition(self):
        without = _deterministic_composite(0.5, 0.5, 0.5, "crypto")
        with_none = _deterministic_composite(0.5, 0.5, 0.5, "crypto", ign=None)
        assert without == with_none


# ---------------------------------------------------------------------------
# compute_signal — integration test with an in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path: Path):
    """Spin up a fresh SQLite DB with the schema applied."""
    db_file = str(tmp_path / "test.db")
    schema = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()
    conn = sqlite3.connect(db_file)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    return db_file


def _insert_prices(db_path: str, symbol: str, asset_class: str, n: int = 50) -> None:
    """Insert synthetic OHLCV rows into prices."""
    conn = sqlite3.connect(db_path)
    for i in range(n):
        ts = 1_700_000_000 + i * 3600
        close = 100.0 + i * 0.5  # steady uptrend
        conn.execute(
            """
            INSERT INTO prices (symbol, asset_class, timeframe, ts, open, high, low, close, volume, funding_rate)
            VALUES (?, ?, '1h', ?, ?, ?, ?, ?, 1000.0, ?)
            """,
            (symbol, asset_class, ts, close, close + 1, close - 1, close, 0.0001),
        )
    conn.commit()
    conn.close()


class TestComputeSignal:
    def test_returns_none_when_no_data(self, temp_db: str):
        import database.db as db_module

        orig_db = db_module._db

        class _FakeDB:
            def connect(self):
                c = sqlite3.connect(temp_db)
                c.row_factory = sqlite3.Row
                return c

        db_module._db = _FakeDB()
        try:
            result = compute_signal("MISSING/USDT", "crypto")
            assert result is None
        finally:
            db_module._db = orig_db

    def test_returns_signal_result_with_data(self, temp_db: str):
        import database.db as db_module

        _insert_prices(temp_db, "BTC/USDT", "crypto")
        orig_db = db_module._db

        class _FakeDB:
            def connect(self):
                c = sqlite3.connect(temp_db)
                c.row_factory = sqlite3.Row
                return c

        db_module._db = _FakeDB()
        try:
            result = compute_signal("BTC/USDT", "crypto")
        finally:
            db_module._db = orig_db

        assert isinstance(result, SignalResult)
        assert result.symbol == "BTC/USDT"
        assert result.asset_class == "crypto"
        assert -1.0 <= result.momentum_score <= 1.0
        assert -1.0 <= result.volatility_score <= 1.0
        assert result.funding_score is not None
        assert -1.0 <= result.funding_score <= 1.0
        assert result.ignition_score is not None
        assert -1.0 <= result.ignition_score <= 1.0
        assert -1.0 <= result.composite_score <= 1.0
        assert result.action in ("buy", "sell", "hold")

    def test_stocks_have_no_funding_score(self, temp_db: str):
        import database.db as db_module

        _insert_prices(temp_db, "AAPL", "stocks")
        orig_db = db_module._db

        class _FakeDB:
            def connect(self):
                c = sqlite3.connect(temp_db)
                c.row_factory = sqlite3.Row
                return c

        db_module._db = _FakeDB()
        try:
            result = compute_signal("AAPL", "stocks")
        finally:
            db_module._db = orig_db

        assert result is not None
        assert result.funding_score is None
        assert result.ignition_score is None
