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
