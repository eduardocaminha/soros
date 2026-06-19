"""Central configuration for the soros trading bot.

All tuneable values live here. Runtime secrets are read from environment
variables and never committed. Execution toggles default to OFF so no
live order reaches an exchange until explicitly enabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).parent.resolve()

DB_PATH: str = os.environ.get("DB_PATH", str(_BASE_DIR / "data" / "soros.db"))

# ---------------------------------------------------------------------------
# Exchange credentials (read from environment; empty strings if unset)
# ---------------------------------------------------------------------------

BINANCE_API_KEY: str = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET: str = os.environ.get("BINANCE_SECRET", "")

ALPACA_API_KEY: str = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET: str = os.environ.get("ALPACA_SECRET", "")
# Base URL for Alpaca — paper endpoint by default
ALPACA_BASE_URL: str = os.environ.get(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)

# ---------------------------------------------------------------------------
# Execution toggles — default OFF (paper trading until explicitly enabled)
# ---------------------------------------------------------------------------

# When False: crypto orders are simulated (dry_run). Never flip to True before
# 48 h+ of validated paper trading.
CRYPTO_LIVE: bool = os.environ.get("CRYPTO_LIVE", "false").lower() == "true"

# When False: stock orders are simulated. Same 48 h gate as crypto.
STOCKS_LIVE: bool = os.environ.get("STOCKS_LIVE", "false").lower() == "true"

# When False: sentiment runner is skipped; bot runs on deterministic signals
# only. Flip to True only after verifying Claude subscription access.
SENTIMENT_ENABLED: bool = (
    os.environ.get("SENTIMENT_ENABLED", "false").lower() == "true"
)

# ---------------------------------------------------------------------------
# Symbols to trade
# ---------------------------------------------------------------------------

# Optional override: when set, these symbols are always included alongside the
# autonomous market-cap base tier.  Leave empty (default) to let the universe
# be fully determined by MARKETCAP_TOP_N + gem scanner.
CRYPTO_SYMBOLS: list[str] = [
    symbol.strip()
    for symbol in os.environ.get("CRYPTO_SYMBOLS", "").split(",")
    if symbol.strip()
]

# Vazio por default: o bot e cripto-only. Para operar acoes (B3 via yfinance
# .SA, ou US via Alpaca), preencha STOCK_SYMBOLS no ambiente.
STOCK_SYMBOLS: list[str] = [
    symbol.strip()
    for symbol in os.environ.get("STOCK_SYMBOLS", "").split(",")
    if symbol.strip()
]

# Watchlist — additional candidates considered by the screener.
# Empty by default; only used when SCREENER_ENABLED=true.
CRYPTO_WATCHLIST: list[str] = [
    symbol.strip()
    for symbol in os.environ.get("CRYPTO_WATCHLIST", "").split(",")
    if symbol.strip()
]

STOCK_WATCHLIST: list[str] = [
    symbol.strip()
    for symbol in os.environ.get("STOCK_WATCHLIST", "").split(",")
    if symbol.strip()
]

# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

# When False (default): only pinned symbols are operated; watchlist is ignored.
# When True: screener ranks pinned ∪ watchlist and selects top-N candidates.
SCREENER_ENABLED: bool = (
    os.environ.get("SCREENER_ENABLED", "false").lower() == "true"
)

# Maximum number of screener-selected symbols (excluding pinned, which are
# always included).  Hard-capped at MAX_OPEN_POSITIONS.
SCREENER_TOP_N: int = int(os.environ.get("SCREENER_TOP_N", "3"))

# Minimum 24 h notional volume (USD) a symbol must show to qualify.
# Conservative default; tune per exchange.
SCREENER_MIN_VOLUME_USD: float = float(
    os.environ.get("SCREENER_MIN_VOLUME_USD", "1000000")
)

# ---------------------------------------------------------------------------
# Autonomous universe — market cap base tier (CoinGecko, keyless)
# ---------------------------------------------------------------------------

# Number of top-N coins by market cap included in the base universe.
MARKETCAP_TOP_N: int = int(os.environ.get("MARKETCAP_TOP_N", "20"))

# How often (seconds) the market cap top-N list is refreshed from CoinGecko.
MARKETCAP_REFRESH_SECS: int = int(os.environ.get("MARKETCAP_REFRESH_SECS", "3600"))

# ---------------------------------------------------------------------------
# DEX discovery signals (DexScreener + GeckoTerminal, keyless)
# Boost gem_score for tokens trending on DEX that also have a Binance pair.
# Execution is always CEX-only; DEX signals are discovery hints only.
# ---------------------------------------------------------------------------

# Multiplier applied to gem_score when a candidate is also DEX-trending.
# Set to 1.0 to disable the DEX boost entirely.
DEX_BOOST_MULTIPLIER: float = float(os.environ.get("DEX_BOOST_MULTIPLIER", "1.5"))

# How long (seconds) to cache DEX trending results before refreshing.
DEX_SCAN_CACHE_SECS: int = int(os.environ.get("DEX_SCAN_CACHE_SECS", "300"))

# ---------------------------------------------------------------------------
# Gem scanner — ignition candidates (CEX via ccxt.fetch_tickers)
# ---------------------------------------------------------------------------

# Volume surge multiplier: a symbol must show >= this multiple of its rolling
# average volume to qualify as a gem candidate (plan: >=2x).
GEM_VOLUME_SURGE_MULTIPLIER: float = float(
    os.environ.get("GEM_VOLUME_SURGE_MULTIPLIER", "2.0")
)

# Minimum price rate-of-change (%) over the short window for gem qualification.
GEM_ROC_MIN_PCT: float = float(os.environ.get("GEM_ROC_MIN_PCT", "3.0"))

# Maximum number of gem candidates surfaced by the scanner per cycle.
GEM_TOP_N: int = int(os.environ.get("GEM_TOP_N", "5"))

# Minimum 24 h notional volume (USD) a gem candidate must meet (liquidity floor).
GEM_MIN_VOLUME_USD: float = float(os.environ.get("GEM_MIN_VOLUME_USD", "500000"))

# ---------------------------------------------------------------------------
# Ignition signal weight
# Used by the signal aggregator when the ignition signal is active.
# Set to 0.0 to disable the ignition signal entirely.
# ---------------------------------------------------------------------------

IGNITION_WEIGHT: float = float(os.environ.get("IGNITION_WEIGHT", "0.15"))

# ---------------------------------------------------------------------------
# Gem risk — position sizing + trailing stop
# ---------------------------------------------------------------------------

# Position size fraction for gem-origin positions (smaller than base, default 5 %).
# Must be <= POSITION_SIZE_PCT.  Set equal to disable the distinction.
GEM_POSITION_SIZE_PCT: float = float(
    os.environ.get("GEM_POSITION_SIZE_PCT", "0.05")
)

# Trailing stop distance (fraction, e.g. 0.05 = 5 %) for gem-origin positions.
# Set to 0.0 to disable trailing stops for gems.
GEM_TRAILING_STOP_PCT: float = float(
    os.environ.get("GEM_TRAILING_STOP_PCT", "0.05")
)

# ---------------------------------------------------------------------------
# Optional sentiment API keys (graceful degradation — absent = neutral score)
# ---------------------------------------------------------------------------

FINNHUB_API_KEY: str = os.environ.get("FINNHUB_API_KEY", "")

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

OHLCV_TIMEFRAME: str = "1h"  # ccxt-compatible timeframe string
OHLCV_LIMIT: int = 200  # candles fetched per call (enough for all signal windows)
# Shorter candle window for watchlist-only symbols — enough for signals (≥26 bars)
# without the overhead of the full pinned history.
WATCHLIST_OHLCV_LIMIT: int = int(os.environ.get("WATCHLIST_OHLCV_LIMIT", "50"))

# How often the main loop ticks (seconds)
LOOP_INTERVAL_SECONDS: int = int(os.environ.get("LOOP_INTERVAL_SECONDS", "3600"))

# ---------------------------------------------------------------------------
# Capital & execution costs
# ---------------------------------------------------------------------------

# Starting equity for P&L calculation (paper mode)
INITIAL_CAPITAL: float = float(os.environ.get("INITIAL_CAPITAL", "10000"))

# Fee and slippage applied to paper execution prices per trade (each side)
FEE_PCT: float = float(os.environ.get("FEE_PCT", "0.001"))        # 0.1 %
SLIPPAGE_PCT: float = float(os.environ.get("SLIPPAGE_PCT", "0.0005"))  # 0.05 %

# ---------------------------------------------------------------------------
# Risk manager — hard limits (not toggleable at runtime)
# ---------------------------------------------------------------------------

MAX_DRAWDOWN_PCT: float = 0.15   # 15 % peak-to-trough stops all new orders
MAX_OPEN_POSITIONS: int = 3      # total across both asset classes

# Order sizing: fraction of equity allocated per position
POSITION_SIZE_PCT: float = float(os.environ.get("POSITION_SIZE_PCT", "0.10"))

# ---------------------------------------------------------------------------
# Signal weights by asset class
# crypto: {momentum, volatility, funding, sentiment}
# stocks: {momentum, volatility, sentiment}
# Weights must sum to 1.0 within each class.
# ---------------------------------------------------------------------------

CRYPTO_SIGNAL_WEIGHTS: dict[str, float] = {
    "momentum": 0.35,
    "volatility": 0.25,
    "funding": 0.20,
    "sentiment": 0.20,
}

STOCK_SIGNAL_WEIGHTS: dict[str, float] = {
    "momentum": 0.45,
    "volatility": 0.30,
    "sentiment": 0.25,
}

# Composite score threshold to trigger a buy/sell action (else hold)
SIGNAL_THRESHOLD: float = 0.25

# Sentiment debate is only triggered when the LLM score contradicts the
# deterministic composite, or when |composite_score| < this value.
DEBATE_DIVERGENCE_THRESHOLD: float = 0.10

# ---------------------------------------------------------------------------
# Sentiment runner
# ---------------------------------------------------------------------------

# Maximum age (seconds) of a sentiment signal before it is considered stale
SENTIMENT_MAX_AGE_SECONDS: int = int(
    os.environ.get("SENTIMENT_MAX_AGE_SECONDS", "7200")
)  # 2 h

# ---------------------------------------------------------------------------
# Settings ALLOWLIST — keys editable at runtime via the settings table
# ---------------------------------------------------------------------------
# Deny-by-default: any key NOT in this dict is locked / read-only.
# Execution toggles (CRYPTO_LIVE, STOCKS_LIVE, SENTIMENT_ENABLED) and hard
# risk limits (MAX_DRAWDOWN_PCT, MAX_OPEN_POSITIONS) are intentionally ABSENT
# and can never be written to the settings table.
#
# Each entry: {"type": <python type>, "min": <lower bound|None>, "max": <upper bound|None>}
SETTINGS_ALLOWLIST: dict[str, dict] = {
    "LOOP_INTERVAL_SECONDS":       {"type": int,   "min": 60,    "max": 86400},
    "SIGNAL_THRESHOLD":            {"type": float, "min": 0.0,   "max": 1.0},
    "DEBATE_DIVERGENCE_THRESHOLD": {"type": float, "min": 0.0,   "max": 1.0},
    "SCREENER_ENABLED":            {"type": bool,  "min": None,  "max": None},
    "SCREENER_TOP_N":              {"type": int,   "min": 1,     "max": 20},
    "SCREENER_MIN_VOLUME_USD":     {"type": float, "min": 0.0,   "max": None},
    "MARKETCAP_TOP_N":             {"type": int,   "min": 1,     "max": 200},
    "MARKETCAP_REFRESH_SECS":      {"type": int,   "min": 60,    "max": 86400},
    "DEX_BOOST_MULTIPLIER":        {"type": float, "min": 1.0,   "max": 10.0},
    "DEX_SCAN_CACHE_SECS":         {"type": int,   "min": 60,    "max": 3600},
    "GEM_VOLUME_SURGE_MULTIPLIER": {"type": float, "min": 1.0,   "max": 20.0},
    "GEM_ROC_MIN_PCT":             {"type": float, "min": 0.0,   "max": 50.0},
    "GEM_TOP_N":                   {"type": int,   "min": 1,     "max": 50},
    "GEM_MIN_VOLUME_USD":          {"type": float, "min": 0.0,   "max": None},
    "IGNITION_WEIGHT":             {"type": float, "min": 0.0,   "max": 1.0},
    "GEM_POSITION_SIZE_PCT":       {"type": float, "min": 0.01,  "max": 0.5},
    "GEM_TRAILING_STOP_PCT":       {"type": float, "min": 0.0,   "max": 0.5},
    "POSITION_SIZE_PCT":           {"type": float, "min": 0.01,  "max": 0.5},
    "WATCHLIST_OHLCV_LIMIT":       {"type": int,   "min": 26,    "max": 500},
    "SENTIMENT_MAX_AGE_SECONDS":   {"type": int,   "min": 300,   "max": 86400},
    "INITIAL_CAPITAL":             {"type": float, "min": 100.0, "max": None},
    "FEE_PCT":                     {"type": float, "min": 0.0,   "max": 0.1},
    "SLIPPAGE_PCT":                {"type": float, "min": 0.0,   "max": 0.1},
}

# Keys that are permanently locked (for UI display — read-only / visually flagged)
SETTINGS_LOCKED: frozenset[str] = frozenset({
    "CRYPTO_LIVE",
    "STOCKS_LIVE",
    "SENTIMENT_ENABLED",
    "MAX_DRAWDOWN_PCT",
    "MAX_OPEN_POSITIONS",
})

# Hardcoded defaults for every allowlisted key.  Used by reload_runtime_overrides()
# so the settings-table fallback is stable even after a previous override is deleted.
_TUNABLE_DEFAULTS: dict[str, Any] = {
    "LOOP_INTERVAL_SECONDS":       3600,
    "SIGNAL_THRESHOLD":            0.25,
    "DEBATE_DIVERGENCE_THRESHOLD": 0.10,
    "SCREENER_ENABLED":            False,
    "SCREENER_TOP_N":              3,
    "SCREENER_MIN_VOLUME_USD":     1_000_000.0,
    "MARKETCAP_TOP_N":             20,
    "MARKETCAP_REFRESH_SECS":      3600,
    "DEX_BOOST_MULTIPLIER":        1.5,
    "DEX_SCAN_CACHE_SECS":         300,
    "GEM_VOLUME_SURGE_MULTIPLIER": 2.0,
    "GEM_ROC_MIN_PCT":             3.0,
    "GEM_TOP_N":                   5,
    "GEM_MIN_VOLUME_USD":          500_000.0,
    "IGNITION_WEIGHT":             0.15,
    "GEM_POSITION_SIZE_PCT":       0.05,
    "GEM_TRAILING_STOP_PCT":       0.05,
    "POSITION_SIZE_PCT":           0.10,
    "WATCHLIST_OHLCV_LIMIT":       50,
    "SENTIMENT_MAX_AGE_SECONDS":   7200,
    "INITIAL_CAPITAL":             10_000.0,
    "FEE_PCT":                     0.001,
    "SLIPPAGE_PCT":                0.0005,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Raise ValueError if the current toggle/credential combination is unsafe.

    Called once at bot startup so misconfigurations are caught before the
    main loop runs. Guards:
    - CRYPTO_LIVE=true requires BINANCE_API_KEY + BINANCE_SECRET
    - STOCKS_LIVE=true requires ALPACA_API_KEY + ALPACA_SECRET
    - Signal weights within each class must sum to 1.0 (±0.001 tolerance)
    """
    errors: list[str] = []

    if CRYPTO_LIVE:
        if not BINANCE_API_KEY or not BINANCE_SECRET:
            errors.append(
                "CRYPTO_LIVE=true requires BINANCE_API_KEY and BINANCE_SECRET"
            )

    if STOCKS_LIVE:
        if not ALPACA_API_KEY or not ALPACA_SECRET:
            errors.append(
                "STOCKS_LIVE=true requires ALPACA_API_KEY and ALPACA_SECRET"
            )

    for name, weights in (
        ("CRYPTO_SIGNAL_WEIGHTS", CRYPTO_SIGNAL_WEIGHTS),
        ("STOCK_SIGNAL_WEIGHTS", STOCK_SIGNAL_WEIGHTS),
    ):
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            errors.append(f"{name} must sum to 1.0, got {total:.4f}")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def reload_runtime_overrides() -> None:
    """Re-apply env > settings > default for every key in SETTINGS_ALLOWLIST.

    Call once at the start of each bot cycle.  Never touches locked keys
    (CRYPTO_LIVE, STOCKS_LIVE, SENTIMENT_ENABLED, MAX_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS) — those remain fixed at their startup values.
    """
    from database import settings_store  # late import avoids circular dependency at startup

    _mod = sys.modules[__name__]
    for key, default in _TUNABLE_DEFAULTS.items():
        env_raw: str | None = os.environ.get(key)
        new_val = settings_store.resolve(key, env_raw, default)
        setattr(_mod, key, new_val)
