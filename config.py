"""Central configuration for the soros trading bot.

All tuneable values live here. Runtime secrets are read from environment
variables and never committed. Execution toggles default to OFF so no
live order reaches an exchange until explicitly enabled.
"""

from __future__ import annotations

import os
from pathlib import Path

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

# Pinned symbols — always included in every cycle.
CRYPTO_SYMBOLS: list[str] = [
    symbol.strip()
    for symbol in os.environ.get(
        "CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT"
    ).split(",")
    if symbol.strip()
]

STOCK_SYMBOLS: list[str] = [
    symbol.strip()
    for symbol in os.environ.get("STOCK_SYMBOLS", "AAPL,MSFT,NVDA").split(",")
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
