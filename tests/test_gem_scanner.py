"""Tests for data/gem_scanner.py — CEX ignition scanner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import config
import data.gem_scanner as gem_module
from data.gem_scanner import (
    GemCandidate,
    _STABLE_EXCLUDE,
    _is_excluded,
    reset_history,
    scan_gems,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker(
    quote_volume: float = 1_000_000.0,
    percentage: float = 5.0,
) -> dict:
    return {"quoteVolume": quote_volume, "percentage": percentage}


def _make_exchange(tickers: dict) -> MagicMock:
    ex = MagicMock()
    ex.fetch_tickers.return_value = tickers
    return ex


@pytest.fixture(autouse=True)
def _clear_history():
    reset_history()
    yield
    reset_history()


# ---------------------------------------------------------------------------
# _is_excluded
# ---------------------------------------------------------------------------

class TestIsExcluded:
    @pytest.mark.parametrize("token", list(_STABLE_EXCLUDE))
    def test_stablecoins_excluded(self, token: str):
        assert _is_excluded(token)

    @pytest.mark.parametrize("token", ["BTCUP", "ETHDOWN", "BNBBULL", "SOLBEAR"])
    def test_direction_leveraged_excluded(self, token: str):
        assert _is_excluded(token)

    @pytest.mark.parametrize("token", ["BTC3L", "ETH3S", "BNB2L", "SOL5S"])
    def test_numeric_leveraged_excluded(self, token: str):
        assert _is_excluded(token)

    @pytest.mark.parametrize("token", ["BTC", "ETH", "SOL", "AVAX", "DOGE"])
    def test_normal_tokens_not_excluded(self, token: str):
        assert not _is_excluded(token)

    def test_bull_suffix_excluded(self):
        assert _is_excluded("LINKBULL")

    def test_bear_suffix_excluded(self):
        assert _is_excluded("LINKBEAR")


# ---------------------------------------------------------------------------
# scan_gems — cold start (no history)
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_returns_empty_on_first_scan(self):
        tickers = {"SOL/USDT": _make_ticker(2_000_000.0, 10.0)}
        ex = _make_exchange(tickers)
        result = scan_gems(exchange=ex)
        assert result == []

    def test_populates_history_on_first_scan(self):
        tickers = {"SOL/USDT": _make_ticker(2_000_000.0, 10.0)}
        ex = _make_exchange(tickers)
        scan_gems(exchange=ex)
        # Second scan: now we have a baseline, so a 2× surge should be detected
        ex2 = _make_exchange({"SOL/USDT": _make_ticker(5_000_000.0, 10.0)})
        result = scan_gems(exchange=ex2)
        assert len(result) == 1
        assert result[0].symbol == "SOL/USDT"


# ---------------------------------------------------------------------------
# scan_gems — filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def _seed_history(self, symbol: str, volume: float):
        """Pre-populate history so subsequent calls have a baseline."""
        tickers = {symbol: _make_ticker(volume, 0.0)}
        scan_gems(exchange=_make_exchange(tickers))

    def test_filters_non_usdt_pairs(self):
        self._seed_history("SOL/BTC", 500_000.0)
        tickers = {"SOL/BTC": _make_ticker(2_000_000.0, 10.0)}
        result = scan_gems(exchange=_make_exchange(tickers))
        assert result == []

    def test_filters_stablecoins(self):
        for stable in ["USDT/USDT", "USDC/USDT", "DAI/USDT"]:
            reset_history()
            self._seed_history(stable, 1_000_000.0)
            tickers = {stable: _make_ticker(5_000_000.0, 10.0)}
            result = scan_gems(exchange=_make_exchange(tickers))
            assert result == [], f"expected {stable} to be excluded"

    def test_filters_leveraged_tokens(self, monkeypatch):
        for lev in ["BTCUP/USDT", "ETHDOWN/USDT", "BTC3L/USDT", "SOL3S/USDT"]:
            reset_history()
            self._seed_history(lev, 1_000_000.0)
            tickers = {lev: _make_ticker(5_000_000.0, 15.0)}
            result = scan_gems(exchange=_make_exchange(tickers))
            assert result == [], f"expected {lev} to be excluded"

    def test_filters_base_symbols(self):
        self._seed_history("BTC/USDT", 1_000_000.0)
        tickers = {"BTC/USDT": _make_ticker(5_000_000.0, 10.0)}
        result = scan_gems(exchange=_make_exchange(tickers), base_symbols={"BTC/USDT"})
        assert result == []

    def test_filters_below_volume_floor(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 1_000_000.0)
        self._seed_history("XYZ/USDT", 400_000.0)
        tickers = {"XYZ/USDT": _make_ticker(800_000.0, 10.0)}
        result = scan_gems(exchange=_make_exchange(tickers))
        assert result == []

    def test_filters_below_roc_minimum(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        self._seed_history("XYZ/USDT", 1_000_000.0)
        tickers = {"XYZ/USDT": _make_ticker(5_000_000.0, 2.9)}
        result = scan_gems(exchange=_make_exchange(tickers))
        assert result == []

    def test_filters_below_surge_multiplier(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        self._seed_history("XYZ/USDT", 2_000_000.0)
        # Surge ratio = 3_000_000 / 2_000_000 = 1.5 < 2.0
        tickers = {"XYZ/USDT": _make_ticker(3_000_000.0, 5.0)}
        result = scan_gems(exchange=_make_exchange(tickers))
        assert result == []


# ---------------------------------------------------------------------------
# scan_gems — gem detection
# ---------------------------------------------------------------------------

class TestGemDetection:
    def _seed(self, tickers: dict):
        scan_gems(exchange=_make_exchange(tickers))

    def test_detects_valid_gem(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 5.0)})
        # 3 000 000 / 1 000 000 = 3x surge
        result = scan_gems(exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}))
        assert len(result) == 1
        gem = result[0]
        assert gem.symbol == "GEM/USDT"
        assert abs(gem.volume_surge_ratio - 3.0) < 1e-6
        assert gem.roc_pct == 5.0
        assert gem.volume_usd_24h == 3_000_000.0

    def test_gem_score_is_surge_times_roc(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        result = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(4_000_000.0, 10.0)})
        )
        assert len(result) == 1
        # surge = 4.0, roc = 10.0 → score = 40.0
        assert abs(result[0].gem_score - 40.0) < 1e-6

    def test_results_ranked_by_gem_score(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "GEM_TOP_N", 10)
        # Seed baselines
        self._seed({
            "A/USDT": _make_ticker(1_000_000.0, 0.0),
            "B/USDT": _make_ticker(1_000_000.0, 0.0),
        })
        # A: surge=3x, roc=5 → score=15; B: surge=2x, roc=10 → score=20
        tickers = {
            "A/USDT": _make_ticker(3_000_000.0, 5.0),
            "B/USDT": _make_ticker(2_000_000.0, 10.0),
        }
        result = scan_gems(exchange=_make_exchange(tickers))
        assert len(result) == 2
        assert result[0].symbol == "B/USDT"
        assert result[1].symbol == "A/USDT"

    def test_top_n_limits_output(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "GEM_TOP_N", 2)
        seed = {f"T{i}/USDT": _make_ticker(1_000_000.0, 0.0) for i in range(5)}
        self._seed(seed)
        surge = {f"T{i}/USDT": _make_ticker(4_000_000.0, 5.0) for i in range(5)}
        result = scan_gems(exchange=_make_exchange(surge))
        assert len(result) == 2

    def test_base_symbols_excluded_from_gems(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        self._seed({"BTC/USDT": _make_ticker(1_000_000.0, 0.0), "GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        tickers = {
            "BTC/USDT": _make_ticker(5_000_000.0, 10.0),
            "GEM/USDT": _make_ticker(5_000_000.0, 10.0),
        }
        result = scan_gems(exchange=_make_exchange(tickers), base_symbols={"BTC/USDT"})
        symbols = [c.symbol for c in result]
        assert "BTC/USDT" not in symbols
        assert "GEM/USDT" in symbols


# ---------------------------------------------------------------------------
# scan_gems — rolling history accumulates
# ---------------------------------------------------------------------------

class TestRollingHistory:
    def test_rolling_avg_uses_multiple_prior_scans(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        # Seed three scans at 1M each → avg = 1M
        for _ in range(3):
            scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(1_000_000.0, 0.0)}))
        # 2.1M / 1M ≈ 2.1 → above threshold
        result = scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(2_100_000.0, 5.0)}))
        assert len(result) == 1
        assert result[0].symbol == "X/USDT"
        assert result[0].volume_surge_ratio > 2.0

    def test_history_capped_at_max_len(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        # Seed 10 scans at 1M → maxlen=5 so only last 5 count
        for _ in range(10):
            scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(1_000_000.0, 0.0)}))
        hist = gem_module._volume_history.get("X/USDT")
        assert hist is not None
        assert len(hist) <= gem_module._VOLUME_HISTORY_MAXLEN


# ---------------------------------------------------------------------------
# scan_gems — error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_returns_empty_on_exchange_error(self):
        ex = MagicMock()
        ex.fetch_tickers.side_effect = Exception("network error")
        result = scan_gems(exchange=ex)
        assert result == []

    def test_handles_missing_quote_volume(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 0.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 0.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        # First scan with None quoteVolume
        scan_gems(exchange=_make_exchange({"X/USDT": {"quoteVolume": None, "percentage": 5.0}}))
        # Second scan: still 0.0 avg → surge won't trigger (0/0)
        result = scan_gems(exchange=_make_exchange({"X/USDT": {"quoteVolume": None, "percentage": 5.0}}))
        assert result == []

    def test_handles_missing_percentage(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        scan_gems(exchange=_make_exchange({"X/USDT": {"quoteVolume": 1_000_000.0, "percentage": None}}))
        # percentage defaults to 0.0, so ROC filter should reject it
        result = scan_gems(exchange=_make_exchange({"X/USDT": {"quoteVolume": 5_000_000.0, "percentage": None}}))
        assert result == []


# ---------------------------------------------------------------------------
# scan_gems — DEX boost
# ---------------------------------------------------------------------------

class TestDexBoost:
    def _seed(self, tickers: dict):
        scan_gems(exchange=_make_exchange(tickers))

    def test_dex_boost_multiplies_score(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "DEX_BOOST_MULTIPLIER", 2.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        # surge=3x, roc=5 → raw_score=15; with 2x DEX boost → 30
        result = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}),
            dex_trending_symbols=frozenset({"GEM"}),
        )
        assert len(result) == 1
        assert abs(result[0].gem_score - 30.0) < 1e-6
        assert result[0].dex_boost is True

    def test_no_dex_set_leaves_score_unchanged(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "DEX_BOOST_MULTIPLIER", 2.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        result = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}),
        )
        assert len(result) == 1
        assert abs(result[0].gem_score - 15.0) < 1e-6
        assert result[0].dex_boost is False

    def test_token_not_in_dex_set_unaffected(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "DEX_BOOST_MULTIPLIER", 2.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        result = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}),
            dex_trending_symbols=frozenset({"OTHER"}),
        )
        assert len(result) == 1
        assert abs(result[0].gem_score - 15.0) < 1e-6
        assert result[0].dex_boost is False

    def test_dex_boost_affects_ranking(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "DEX_BOOST_MULTIPLIER", 3.0)
        monkeypatch.setattr(config, "GEM_TOP_N", 10)
        self._seed({
            "A/USDT": _make_ticker(1_000_000.0, 0.0),
            "B/USDT": _make_ticker(1_000_000.0, 0.0),
        })
        # A: surge=3x, roc=5 → raw=15; no DEX → 15
        # B: surge=2x, roc=4 → raw=8; DEX 3x → 24
        tickers = {
            "A/USDT": _make_ticker(3_000_000.0, 5.0),
            "B/USDT": _make_ticker(2_000_000.0, 4.0),
        }
        result = scan_gems(
            exchange=_make_exchange(tickers),
            dex_trending_symbols=frozenset({"B"}),
        )
        assert result[0].symbol == "B/USDT"
        assert result[0].dex_boost is True
        assert result[1].symbol == "A/USDT"
        assert result[1].dex_boost is False

    def test_multiplier_one_is_noop(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        monkeypatch.setattr(config, "DEX_BOOST_MULTIPLIER", 1.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        result_with_dex = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}),
            dex_trending_symbols=frozenset({"GEM"}),
        )
        reset_history()
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        result_without_dex = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}),
        )
        assert abs(result_with_dex[0].gem_score - result_without_dex[0].gem_score) < 1e-6

    def test_dex_boost_flag_false_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        self._seed({"GEM/USDT": _make_ticker(1_000_000.0, 0.0)})
        result = scan_gems(
            exchange=_make_exchange({"GEM/USDT": _make_ticker(3_000_000.0, 5.0)}),
        )
        assert result[0].dex_boost is False


# ---------------------------------------------------------------------------
# reset_history
# ---------------------------------------------------------------------------

class TestResetHistory:
    def test_reset_clears_history(self):
        scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(1_000_000.0, 5.0)}))
        assert "X/USDT" in gem_module._volume_history
        reset_history()
        assert gem_module._volume_history == {}

    def test_after_reset_cold_start_again(self, monkeypatch):
        monkeypatch.setattr(config, "GEM_MIN_VOLUME_USD", 500_000.0)
        monkeypatch.setattr(config, "GEM_ROC_MIN_PCT", 3.0)
        monkeypatch.setattr(config, "GEM_VOLUME_SURGE_MULTIPLIER", 2.0)
        scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(1_000_000.0, 0.0)}))
        scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(5_000_000.0, 10.0)}))
        reset_history()
        # After reset, first scan returns nothing again
        result = scan_gems(exchange=_make_exchange({"X/USDT": _make_ticker(5_000_000.0, 10.0)}))
        assert result == []
