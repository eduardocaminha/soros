"""Tests for config execution toggles and validate_config()."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

import config


class TestTogglesDefault:
    def test_crypto_live_default_false(self, monkeypatch):
        monkeypatch.delenv("CRYPTO_LIVE", raising=False)
        # Reload to pick up env change
        importlib.reload(config)
        assert config.CRYPTO_LIVE is False

    def test_stocks_live_default_false(self, monkeypatch):
        monkeypatch.delenv("STOCKS_LIVE", raising=False)
        importlib.reload(config)
        assert config.STOCKS_LIVE is False

    def test_sentiment_enabled_default_false(self, monkeypatch):
        monkeypatch.delenv("SENTIMENT_ENABLED", raising=False)
        importlib.reload(config)
        assert config.SENTIMENT_ENABLED is False

    def test_crypto_live_true_from_env(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_LIVE", "true")
        importlib.reload(config)
        assert config.CRYPTO_LIVE is True

    def test_stocks_live_true_from_env(self, monkeypatch):
        monkeypatch.setenv("STOCKS_LIVE", "true")
        importlib.reload(config)
        assert config.STOCKS_LIVE is True

    def test_sentiment_enabled_true_from_env(self, monkeypatch):
        monkeypatch.setenv("SENTIMENT_ENABLED", "true")
        importlib.reload(config)
        assert config.SENTIMENT_ENABLED is True

    def test_toggle_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_LIVE", "TRUE")
        importlib.reload(config)
        assert config.CRYPTO_LIVE is True

    def teardown_method(self):
        # Always reload back to clean defaults after each test
        for key in ("CRYPTO_LIVE", "STOCKS_LIVE", "SENTIMENT_ENABLED"):
            os.environ.pop(key, None)
        importlib.reload(config)


class TestValidateConfig:
    def test_all_off_no_credentials_passes(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        monkeypatch.setattr(config, "BINANCE_API_KEY", "")
        monkeypatch.setattr(config, "BINANCE_SECRET", "")
        monkeypatch.setattr(config, "ALPACA_API_KEY", "")
        monkeypatch.setattr(config, "ALPACA_SECRET", "")
        config.validate_config()  # must not raise

    def test_crypto_live_requires_binance_key(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        monkeypatch.setattr(config, "BINANCE_API_KEY", "")
        monkeypatch.setattr(config, "BINANCE_SECRET", "")
        with pytest.raises(ValueError, match="BINANCE_API_KEY"):
            config.validate_config()

    def test_crypto_live_requires_binance_secret(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        monkeypatch.setattr(config, "BINANCE_API_KEY", "somekey")
        monkeypatch.setattr(config, "BINANCE_SECRET", "")
        with pytest.raises(ValueError, match="BINANCE"):
            config.validate_config()

    def test_crypto_live_with_credentials_passes(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        monkeypatch.setattr(config, "BINANCE_API_KEY", "key")
        monkeypatch.setattr(config, "BINANCE_SECRET", "secret")
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        config.validate_config()

    def test_stocks_live_requires_alpaca_key(self, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        monkeypatch.setattr(config, "ALPACA_API_KEY", "")
        monkeypatch.setattr(config, "ALPACA_SECRET", "")
        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            config.validate_config()

    def test_stocks_live_requires_alpaca_secret(self, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        monkeypatch.setattr(config, "ALPACA_API_KEY", "somekey")
        monkeypatch.setattr(config, "ALPACA_SECRET", "")
        with pytest.raises(ValueError, match="ALPACA"):
            config.validate_config()

    def test_stocks_live_with_credentials_passes(self, monkeypatch):
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        monkeypatch.setattr(config, "ALPACA_API_KEY", "key")
        monkeypatch.setattr(config, "ALPACA_SECRET", "secret")
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        config.validate_config()

    def test_invalid_crypto_weights_raises(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        monkeypatch.setattr(config, "CRYPTO_SIGNAL_WEIGHTS", {
            "momentum": 0.5,
            "volatility": 0.5,
            "funding": 0.5,
            "sentiment": 0.5,
        })
        with pytest.raises(ValueError, match="CRYPTO_SIGNAL_WEIGHTS"):
            config.validate_config()

    def test_invalid_stock_weights_raises(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        monkeypatch.setattr(config, "STOCK_SIGNAL_WEIGHTS", {
            "momentum": 0.5,
            "volatility": 0.5,
            "sentiment": 0.5,
        })
        with pytest.raises(ValueError, match="STOCK_SIGNAL_WEIGHTS"):
            config.validate_config()

    def test_default_weights_are_valid(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", False)
        monkeypatch.setattr(config, "STOCKS_LIVE", False)
        config.validate_config()  # default weights must sum to 1.0

    def test_multiple_errors_reported_together(self, monkeypatch):
        monkeypatch.setattr(config, "CRYPTO_LIVE", True)
        monkeypatch.setattr(config, "STOCKS_LIVE", True)
        monkeypatch.setattr(config, "BINANCE_API_KEY", "")
        monkeypatch.setattr(config, "BINANCE_SECRET", "")
        monkeypatch.setattr(config, "ALPACA_API_KEY", "")
        monkeypatch.setattr(config, "ALPACA_SECRET", "")
        with pytest.raises(ValueError) as exc_info:
            config.validate_config()
        msg = str(exc_info.value)
        assert "BINANCE" in msg
        assert "ALPACA" in msg


class TestWatchlistAndScreener:
    def teardown_method(self):
        for key in (
            "CRYPTO_WATCHLIST",
            "STOCK_WATCHLIST",
            "SCREENER_ENABLED",
            "SCREENER_TOP_N",
            "SCREENER_MIN_VOLUME_USD",
            "FINNHUB_API_KEY",
        ):
            os.environ.pop(key, None)
        importlib.reload(config)

    def test_watchlists_default_empty(self, monkeypatch):
        monkeypatch.delenv("CRYPTO_WATCHLIST", raising=False)
        monkeypatch.delenv("STOCK_WATCHLIST", raising=False)
        importlib.reload(config)
        assert config.CRYPTO_WATCHLIST == []
        assert config.STOCK_WATCHLIST == []

    def test_crypto_watchlist_parsed_from_env(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_WATCHLIST", "DOGE/USDT, ADA/USDT, AVAX/USDT")
        importlib.reload(config)
        assert config.CRYPTO_WATCHLIST == ["DOGE/USDT", "ADA/USDT", "AVAX/USDT"]

    def test_stock_watchlist_parsed_from_env(self, monkeypatch):
        monkeypatch.setenv("STOCK_WATCHLIST", "TSLA,AMZN,GOOG")
        importlib.reload(config)
        assert config.STOCK_WATCHLIST == ["TSLA", "AMZN", "GOOG"]

    def test_screener_enabled_default_false(self, monkeypatch):
        monkeypatch.delenv("SCREENER_ENABLED", raising=False)
        importlib.reload(config)
        assert config.SCREENER_ENABLED is False

    def test_screener_enabled_true_from_env(self, monkeypatch):
        monkeypatch.setenv("SCREENER_ENABLED", "true")
        importlib.reload(config)
        assert config.SCREENER_ENABLED is True

    def test_screener_enabled_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SCREENER_ENABLED", "TRUE")
        importlib.reload(config)
        assert config.SCREENER_ENABLED is True

    def test_screener_top_n_default(self, monkeypatch):
        monkeypatch.delenv("SCREENER_TOP_N", raising=False)
        importlib.reload(config)
        assert config.SCREENER_TOP_N == 3

    def test_screener_top_n_from_env(self, monkeypatch):
        monkeypatch.setenv("SCREENER_TOP_N", "5")
        importlib.reload(config)
        assert config.SCREENER_TOP_N == 5

    def test_screener_min_volume_default(self, monkeypatch):
        monkeypatch.delenv("SCREENER_MIN_VOLUME_USD", raising=False)
        importlib.reload(config)
        assert config.SCREENER_MIN_VOLUME_USD == 1_000_000.0

    def test_screener_min_volume_from_env(self, monkeypatch):
        monkeypatch.setenv("SCREENER_MIN_VOLUME_USD", "500000")
        importlib.reload(config)
        assert config.SCREENER_MIN_VOLUME_USD == 500_000.0

    def test_finnhub_key_default_empty(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        importlib.reload(config)
        assert config.FINNHUB_API_KEY == ""

    def test_finnhub_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "xyz789")
        importlib.reload(config)
        assert config.FINNHUB_API_KEY == "xyz789"

    def test_watchlist_empty_string_excluded(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_WATCHLIST", "BTC/USDT,,ETH/USDT,")
        importlib.reload(config)
        assert config.CRYPTO_WATCHLIST == ["BTC/USDT", "ETH/USDT"]


class TestCryptoSymbolsOptionalOverride:
    def teardown_method(self):
        os.environ.pop("CRYPTO_SYMBOLS", None)
        importlib.reload(config)

    def test_crypto_symbols_default_empty(self, monkeypatch):
        monkeypatch.delenv("CRYPTO_SYMBOLS", raising=False)
        importlib.reload(config)
        assert config.CRYPTO_SYMBOLS == []

    def test_crypto_symbols_override_from_env(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT")
        importlib.reload(config)
        assert config.CRYPTO_SYMBOLS == ["BTC/USDT", "ETH/USDT"]

    def test_crypto_symbols_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_SYMBOLS", " BTC/USDT , SOL/USDT ")
        importlib.reload(config)
        assert config.CRYPTO_SYMBOLS == ["BTC/USDT", "SOL/USDT"]

    def test_crypto_symbols_filters_empty_entries(self, monkeypatch):
        monkeypatch.setenv("CRYPTO_SYMBOLS", "BTC/USDT,,ETH/USDT,")
        importlib.reload(config)
        assert config.CRYPTO_SYMBOLS == ["BTC/USDT", "ETH/USDT"]


class TestAutonomousUniverseConfig:
    def teardown_method(self):
        for key in (
            "MARKETCAP_TOP_N",
            "MARKETCAP_REFRESH_SECS",
            "GEM_VOLUME_SURGE_MULTIPLIER",
            "GEM_ROC_MIN_PCT",
            "GEM_TOP_N",
            "GEM_MIN_VOLUME_USD",
            "IGNITION_WEIGHT",
            "GEM_TRAILING_STOP_PCT",
        ):
            os.environ.pop(key, None)
        importlib.reload(config)

    def test_marketcap_top_n_default(self, monkeypatch):
        monkeypatch.delenv("MARKETCAP_TOP_N", raising=False)
        importlib.reload(config)
        assert config.MARKETCAP_TOP_N == 20

    def test_marketcap_top_n_from_env(self, monkeypatch):
        monkeypatch.setenv("MARKETCAP_TOP_N", "50")
        importlib.reload(config)
        assert config.MARKETCAP_TOP_N == 50

    def test_marketcap_refresh_secs_default(self, monkeypatch):
        monkeypatch.delenv("MARKETCAP_REFRESH_SECS", raising=False)
        importlib.reload(config)
        assert config.MARKETCAP_REFRESH_SECS == 3600

    def test_marketcap_refresh_secs_from_env(self, monkeypatch):
        monkeypatch.setenv("MARKETCAP_REFRESH_SECS", "1800")
        importlib.reload(config)
        assert config.MARKETCAP_REFRESH_SECS == 1800

    def test_gem_volume_surge_multiplier_default(self, monkeypatch):
        monkeypatch.delenv("GEM_VOLUME_SURGE_MULTIPLIER", raising=False)
        importlib.reload(config)
        assert config.GEM_VOLUME_SURGE_MULTIPLIER == 2.0

    def test_gem_volume_surge_multiplier_from_env(self, monkeypatch):
        monkeypatch.setenv("GEM_VOLUME_SURGE_MULTIPLIER", "3.5")
        importlib.reload(config)
        assert config.GEM_VOLUME_SURGE_MULTIPLIER == 3.5

    def test_gem_roc_min_pct_default(self, monkeypatch):
        monkeypatch.delenv("GEM_ROC_MIN_PCT", raising=False)
        importlib.reload(config)
        assert config.GEM_ROC_MIN_PCT == 3.0

    def test_gem_roc_min_pct_from_env(self, monkeypatch):
        monkeypatch.setenv("GEM_ROC_MIN_PCT", "5.0")
        importlib.reload(config)
        assert config.GEM_ROC_MIN_PCT == 5.0

    def test_gem_top_n_default(self, monkeypatch):
        monkeypatch.delenv("GEM_TOP_N", raising=False)
        importlib.reload(config)
        assert config.GEM_TOP_N == 5

    def test_gem_top_n_from_env(self, monkeypatch):
        monkeypatch.setenv("GEM_TOP_N", "10")
        importlib.reload(config)
        assert config.GEM_TOP_N == 10

    def test_gem_min_volume_usd_default(self, monkeypatch):
        monkeypatch.delenv("GEM_MIN_VOLUME_USD", raising=False)
        importlib.reload(config)
        assert config.GEM_MIN_VOLUME_USD == 500_000.0

    def test_gem_min_volume_usd_from_env(self, monkeypatch):
        monkeypatch.setenv("GEM_MIN_VOLUME_USD", "1000000")
        importlib.reload(config)
        assert config.GEM_MIN_VOLUME_USD == 1_000_000.0

    def test_ignition_weight_default(self, monkeypatch):
        monkeypatch.delenv("IGNITION_WEIGHT", raising=False)
        importlib.reload(config)
        assert config.IGNITION_WEIGHT == 0.15

    def test_ignition_weight_from_env(self, monkeypatch):
        monkeypatch.setenv("IGNITION_WEIGHT", "0.20")
        importlib.reload(config)
        assert config.IGNITION_WEIGHT == 0.20

    def test_ignition_weight_zero_disables(self, monkeypatch):
        monkeypatch.setenv("IGNITION_WEIGHT", "0.0")
        importlib.reload(config)
        assert config.IGNITION_WEIGHT == 0.0

    def test_gem_trailing_stop_pct_default(self, monkeypatch):
        monkeypatch.delenv("GEM_TRAILING_STOP_PCT", raising=False)
        importlib.reload(config)
        assert config.GEM_TRAILING_STOP_PCT == 0.05

    def test_gem_trailing_stop_pct_from_env(self, monkeypatch):
        monkeypatch.setenv("GEM_TRAILING_STOP_PCT", "0.08")
        importlib.reload(config)
        assert config.GEM_TRAILING_STOP_PCT == 0.08

    def test_gem_trailing_stop_pct_zero_disables(self, monkeypatch):
        monkeypatch.setenv("GEM_TRAILING_STOP_PCT", "0.0")
        importlib.reload(config)
        assert config.GEM_TRAILING_STOP_PCT == 0.0


class TestDexDiscoveryConfig:
    def teardown_method(self):
        for key in ("DEX_BOOST_MULTIPLIER", "DEX_SCAN_CACHE_SECS"):
            os.environ.pop(key, None)
        importlib.reload(config)

    def test_dex_boost_multiplier_default(self, monkeypatch):
        monkeypatch.delenv("DEX_BOOST_MULTIPLIER", raising=False)
        importlib.reload(config)
        assert config.DEX_BOOST_MULTIPLIER == 1.5

    def test_dex_boost_multiplier_from_env(self, monkeypatch):
        monkeypatch.setenv("DEX_BOOST_MULTIPLIER", "2.0")
        importlib.reload(config)
        assert config.DEX_BOOST_MULTIPLIER == 2.0

    def test_dex_boost_multiplier_one_disables_boost(self, monkeypatch):
        monkeypatch.setenv("DEX_BOOST_MULTIPLIER", "1.0")
        importlib.reload(config)
        assert config.DEX_BOOST_MULTIPLIER == 1.0

    def test_dex_scan_cache_secs_default(self, monkeypatch):
        monkeypatch.delenv("DEX_SCAN_CACHE_SECS", raising=False)
        importlib.reload(config)
        assert config.DEX_SCAN_CACHE_SECS == 300

    def test_dex_scan_cache_secs_from_env(self, monkeypatch):
        monkeypatch.setenv("DEX_SCAN_CACHE_SECS", "600")
        importlib.reload(config)
        assert config.DEX_SCAN_CACHE_SECS == 600
