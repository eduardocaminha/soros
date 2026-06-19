"""Deterministic signal computation: momentum + volatility_breakout + funding_rate + ignition.

Reads OHLCV rows from the prices table, computes the four deterministic
signal scores, derives a preliminary composite score (deterministic weights
only, no sentiment), and upserts into the signals table.

The signal_aggregator step will later blend in the sentiment signal to
produce the final composite and action.

Usage:
    from signals.compute import compute_once
    results = compute_once()          # all symbols in config
    results = compute_once(["BTC/USDT"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

import config
from database.db import get_connection, get_logger
from signals import funding, ignition, momentum, volatility

_log = get_logger(__name__)


@dataclass
class SignalResult:
    symbol: str
    asset_class: str
    ts: int
    momentum_score: float
    volatility_score: float
    funding_score: float | None
    composite_score: float
    action: str  # 'buy' | 'sell' | 'hold'
    ignition_score: float | None = None  # None for stocks; added after initial schema


def _load_prices(symbol: str, asset_class: str, limit: int = 200) -> pd.DataFrame:
    """Return the most recent *limit* OHLCV rows for *symbol*."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT ts, open, high, low, close, volume, funding_rate
        FROM prices
        WHERE symbol = ? AND asset_class = ? AND timeframe = ?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (symbol, asset_class, config.OHLCV_TIMEFRAME, limit),
    ).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume", "funding_rate"]
        )

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "funding_rate"])
    return df.sort_values("ts").reset_index(drop=True)


def _deterministic_composite(
    mom: float,
    vol: float,
    fund: float | None,
    asset_class: str,
    *,
    ign: float | None = None,
) -> float:
    """Weighted composite from deterministic signals only (no sentiment).

    Uses config weights but re-normalises them to exclude sentiment, so the
    output is in [-1, 1] regardless of whether sentiment is present.
    Ignition is included when *ign* is provided (crypto only).
    """
    if asset_class == "crypto":
        raw_weights = {
            "momentum": config.CRYPTO_SIGNAL_WEIGHTS["momentum"],
            "volatility": config.CRYPTO_SIGNAL_WEIGHTS["volatility"],
            "funding": config.CRYPTO_SIGNAL_WEIGHTS.get("funding", 0.0),
        }
        scores = {
            "momentum": mom,
            "volatility": vol,
            "funding": fund if fund is not None else 0.0,
        }
        if ign is not None:
            raw_weights["ignition"] = config.IGNITION_WEIGHT
            scores["ignition"] = ign
    else:
        raw_weights = {
            "momentum": config.STOCK_SIGNAL_WEIGHTS["momentum"],
            "volatility": config.STOCK_SIGNAL_WEIGHTS["volatility"],
        }
        scores = {"momentum": mom, "volatility": vol}

    total_weight = sum(raw_weights.values())
    if total_weight == 0.0:
        return 0.0

    composite = sum(raw_weights[k] * scores[k] for k in raw_weights) / total_weight
    return max(-1.0, min(1.0, composite))


def _action(composite: float) -> str:
    if composite >= config.SIGNAL_THRESHOLD:
        return "buy"
    if composite <= -config.SIGNAL_THRESHOLD:
        return "sell"
    return "hold"


def _upsert_signal(result: SignalResult) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO signals
            (symbol, asset_class, ts, momentum_score, volatility_score,
             funding_score, ignition_score, composite_score, action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        (
            result.symbol,
            result.asset_class,
            result.ts,
            result.momentum_score,
            result.volatility_score,
            result.funding_score,
            result.ignition_score,
            result.composite_score,
            result.action,
        ),
    )
    conn.commit()


def compute_signal(symbol: str, asset_class: str = "crypto") -> SignalResult | None:
    """Compute all deterministic signals for *symbol* and persist to DB.

    Returns:
        SignalResult on success, None when there is insufficient price data.
    """
    df = _load_prices(symbol, asset_class)
    if df.empty:
        _log.warning("no price data for %s — skipping signal computation", symbol)
        return None

    closes = df["close"]
    latest_funding = df["funding_rate"].iloc[-1] if asset_class == "crypto" else None

    mom = momentum.compute(closes)
    vol = volatility.compute(df)
    fund = funding.compute(latest_funding)
    ign = ignition.compute(df) if asset_class == "crypto" else None

    composite = _deterministic_composite(
        mom, vol, fund if asset_class == "crypto" else None, asset_class, ign=ign
    )
    action = _action(composite)
    ts = int(time.time())

    result = SignalResult(
        symbol=symbol,
        asset_class=asset_class,
        ts=ts,
        momentum_score=mom,
        volatility_score=vol,
        funding_score=fund if asset_class == "crypto" else None,
        composite_score=composite,
        action=action,
        ignition_score=ign,
    )
    _upsert_signal(result)
    _log.info(
        "signal %s: mom=%.3f vol=%.3f fund=%s ign=%s composite=%.3f action=%s",
        symbol,
        mom,
        vol,
        f"{fund:.3f}" if fund is not None else "n/a",
        f"{ign:.3f}" if ign is not None else "n/a",
        composite,
        action,
    )
    return result


def compute_once(
    crypto_symbols: list[str] | None = None,
    stock_symbols: list[str] | None = None,
) -> list[SignalResult]:
    """Compute signals for all configured symbols.

    Args:
        crypto_symbols: Override crypto symbol list; defaults to config.CRYPTO_SYMBOLS.
        stock_symbols:  Override stock symbol list; defaults to config.STOCK_SYMBOLS.

    Returns:
        List of SignalResult for every symbol that had enough price data.
    """
    results: list[SignalResult] = []

    for sym in crypto_symbols or config.CRYPTO_SYMBOLS:
        r = compute_signal(sym, "crypto")
        if r is not None:
            results.append(r)

    for sym in stock_symbols or config.STOCK_SYMBOLS:
        r = compute_signal(sym, "stocks")
        if r is not None:
            results.append(r)

    return results


if __name__ == "__main__":
    for r in compute_once():
        ign_str = f"{r.ignition_score:+.3f}" if r.ignition_score is not None else "n/a"
        print(
            f"{r.symbol:12s}  mom={r.momentum_score:+.3f}  "
            f"vol={r.volatility_score:+.3f}  "
            f"fund={r.funding_score:+.3f}  "
            f"ign={ign_str}  "
            f"composite={r.composite_score:+.3f}  {r.action}"
        )
