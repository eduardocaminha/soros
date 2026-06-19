"""Tests for data/assembler.py — universe assembler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from data.assembler import AssembledUniverse, assemble_universe
from data.gem_scanner import GemCandidate
from data.binance_symbols import reset_cache as reset_binance_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_network_binance():
    """Disable the Binance symbol filter for all tests by default (returns None)."""
    reset_binance_cache()
    with patch("data.assembler.get_tradeable_symbols", return_value=None):
        yield
    reset_binance_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_base(symbols: list[str]):
    return patch("data.assembler.get_base_universe", return_value=symbols)


def _mock_dex(trending: frozenset[str]):
    return patch("data.assembler.get_dex_trending_symbols", return_value=trending)


def _mock_gems(candidates: list[GemCandidate]):
    return patch("data.assembler.scan_gems", return_value=candidates)


def _mock_tradeable(symbols: frozenset[str] | None):
    return patch("data.assembler.get_tradeable_symbols", return_value=symbols)


def _gem(symbol: str, score: float = 5.0, dex_boost: bool = False) -> GemCandidate:
    return GemCandidate(
        symbol=symbol,
        volume_usd_24h=1_000_000.0,
        volume_surge_ratio=3.0,
        roc_pct=10.0,
        gem_score=score,
        dex_boost=dex_boost,
    )


# ---------------------------------------------------------------------------
# AssembledUniverse structure
# ---------------------------------------------------------------------------

class TestAssembledUniverse:
    def test_base_only_no_gems(self):
        with _mock_base(["BTC/USDT", "ETH/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([]):
            uni = assemble_universe()
        assert uni.base_symbols == ["BTC/USDT", "ETH/USDT"]
        assert uni.gem_symbols == []
        assert uni.all_symbols == ["BTC/USDT", "ETH/USDT"]
        assert uni.origins == {"BTC/USDT": "base", "ETH/USDT": "base"}

    def test_gems_appended_after_base(self):
        with _mock_base(["BTC/USDT", "ETH/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([_gem("XYZ/USDT")]):
            uni = assemble_universe()
        assert uni.all_symbols == ["BTC/USDT", "ETH/USDT", "XYZ/USDT"]

    def test_gem_origin_tagged_correctly(self):
        with _mock_base(["BTC/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([_gem("XYZ/USDT", dex_boost=False)]):
            uni = assemble_universe()
        assert uni.origins["XYZ/USDT"] == "gem"

    def test_dex_boosted_gem_origin(self):
        with _mock_base(["BTC/USDT"]), \
             _mock_dex(frozenset({"XYZ"})), \
             _mock_gems([_gem("XYZ/USDT", dex_boost=True)]):
            uni = assemble_universe()
        assert uni.origins["XYZ/USDT"] == "dex_boosted"
        assert "XYZ/USDT" in uni.dex_boosted_symbols

    def test_base_symbols_not_duplicated_by_gems(self):
        with _mock_base(["BTC/USDT", "ETH/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([_gem("ETH/USDT")]):  # gem same as base
            uni = assemble_universe()
        # gem_scanner receives base_set and excludes them; assembler also
        # dedups, but scan_gems should already have excluded it
        assert uni.all_symbols.count("ETH/USDT") == 1
        assert uni.origins["ETH/USDT"] == "base"

    def test_returns_assembled_universe_type(self):
        with _mock_base([]), _mock_dex(frozenset()), _mock_gems([]):
            uni = assemble_universe()
        assert isinstance(uni, AssembledUniverse)

    def test_gem_candidates_stored(self):
        gem = _gem("XYZ/USDT")
        with _mock_base(["BTC/USDT"]), _mock_dex(frozenset()), _mock_gems([gem]):
            uni = assemble_universe()
        assert uni.gem_candidates == [gem]

    def test_empty_base_and_gems_gives_empty_universe(self):
        with _mock_base([]), _mock_dex(frozenset()), _mock_gems([]):
            uni = assemble_universe()
        assert uni.all_symbols == []
        assert uni.origins == {}

    def test_multiple_gems_all_tagged(self):
        gems = [_gem("A/USDT", dex_boost=False), _gem("B/USDT", dex_boost=True)]
        with _mock_base([]), _mock_dex(frozenset({"B"})), _mock_gems(gems):
            uni = assemble_universe()
        assert uni.origins["A/USDT"] == "gem"
        assert uni.origins["B/USDT"] == "dex_boosted"
        assert len(uni.all_symbols) == 2

    def test_base_order_preserved(self):
        base = ["C/USDT", "A/USDT", "B/USDT"]
        with _mock_base(base), _mock_dex(frozenset()), _mock_gems([]):
            uni = assemble_universe()
        assert uni.all_symbols[:3] == base

    def test_scan_gems_receives_base_set(self):
        """scan_gems must receive the base set so it can exclude duplicates."""
        base = ["BTC/USDT", "ETH/USDT"]
        with _mock_base(base), \
             _mock_dex(frozenset()), \
             patch("data.assembler.scan_gems", return_value=[]) as mock_scan:
            assemble_universe()
        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs["base_symbols"] == set(base)

    def test_dex_trending_forwarded_to_gem_scanner(self):
        dex = frozenset({"SOL", "AVAX"})
        with _mock_base([]), \
             _mock_dex(dex), \
             patch("data.assembler.scan_gems", return_value=[]) as mock_scan:
            assemble_universe()
        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs["dex_trending_symbols"] == dex


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_empty_base_falls_back_gracefully(self):
        with _mock_base([]), _mock_dex(frozenset()), _mock_gems([_gem("XYZ/USDT")]):
            uni = assemble_universe()
        assert "XYZ/USDT" in uni.all_symbols

    def test_dex_failure_still_returns_universe(self):
        with _mock_base(["BTC/USDT"]), \
             patch("data.assembler.get_dex_trending_symbols", side_effect=OSError("timeout")), \
             _mock_gems([]):
            # dex failure should propagate (caller is responsible for retry);
            # assembler does not swallow it — the dex_scanner itself caches on failure
            with pytest.raises(OSError):
                assemble_universe()


# ---------------------------------------------------------------------------
# Binance symbol filtering
# ---------------------------------------------------------------------------

class TestBinanceSymbolFiltering:
    """Assembler must drop base/gem symbols absent from Binance spot."""

    def test_base_coin_not_on_binance_dropped(self):
        tradeable = frozenset({"BTC/USDT", "ETH/USDT"})
        with _mock_base(["BTC/USDT", "LEO/USDT", "ETH/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([]), \
             _mock_tradeable(tradeable):
            uni = assemble_universe()
        assert "LEO/USDT" not in uni.all_symbols
        assert uni.base_symbols == ["BTC/USDT", "ETH/USDT"]

    def test_gem_not_on_binance_dropped(self):
        tradeable = frozenset({"BTC/USDT", "ETH/USDT"})
        with _mock_base(["BTC/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([_gem("ETH/USDT"), _gem("OFFCHAIN/USDT")]), \
             _mock_tradeable(tradeable):
            uni = assemble_universe()
        assert "OFFCHAIN/USDT" not in uni.all_symbols
        assert "ETH/USDT" in uni.all_symbols

    def test_filter_unavailable_passes_everything(self):
        # When tradeable is None (markets never loaded), nothing is dropped.
        with _mock_base(["BTC/USDT", "LEO/USDT"]), \
             _mock_dex(frozenset()), \
             _mock_gems([_gem("RAIN/USDT")]), \
             _mock_tradeable(None):
            uni = assemble_universe()
        assert "LEO/USDT" in uni.all_symbols
        assert "RAIN/USDT" in uni.all_symbols
