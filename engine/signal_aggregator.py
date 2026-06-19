"""Signal aggregator — blends sentiment (4th signal) into the deterministic composite.

Reads the latest row from the signals table for each symbol, fetches the
current sentiment score, re-weights all signals according to class-specific
config weights, and updates the row's composite_score and action in place.

Step 14 of plan pl-b0cd.  Called after signals.compute.compute_once() so the
deterministic scores already exist in the DB.

Usage:
    from engine.signal_aggregator import aggregate_once
    results = aggregate_once()
    results = aggregate_once(["BTC/USDT"], ["AAPL"])
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from database.db import get_connection, get_logger
from signals import sentiment as sentiment_signal

_log = get_logger(__name__)


@dataclass
class AggregatedSignal:
    symbol: str
    asset_class: str
    signal_id: int
    momentum_score: float
    volatility_score: float
    funding_score: float | None
    sentiment_score: float
    composite_score: float
    action: str  # 'buy' | 'sell' | 'hold'
    ignition_score: float | None = None  # None for stocks; added after initial schema


def _final_composite(
    mom: float,
    vol: float,
    fund: float | None,
    sent: float,
    asset_class: str,
    *,
    ign: float | None = None,
) -> float:
    """Weighted composite across all signals using class-specific config weights.

    Ignition is included when *ign* is provided (crypto only), adding
    config.IGNITION_WEIGHT to the pool before re-normalisation.
    """
    if asset_class == "crypto":
        base_weights = config.CRYPTO_SIGNAL_WEIGHTS
        scores = {
            "momentum": mom,
            "volatility": vol,
            "funding": fund if fund is not None else 0.0,
            "sentiment": sent,
        }
        if ign is not None:
            weights = {**base_weights, "ignition": config.IGNITION_WEIGHT}
            scores["ignition"] = ign
        else:
            weights = base_weights
    else:
        weights = config.STOCK_SIGNAL_WEIGHTS
        scores = {"momentum": mom, "volatility": vol, "sentiment": sent}

    total = sum(weights.get(k, 0.0) for k in scores)
    if total == 0.0:
        return 0.0

    composite = sum(weights.get(k, 0.0) * v for k, v in scores.items()) / total
    return max(-1.0, min(1.0, composite))


def _action(composite: float) -> str:
    if composite >= config.SIGNAL_THRESHOLD:
        return "buy"
    if composite <= -config.SIGNAL_THRESHOLD:
        return "sell"
    return "hold"


def aggregate_signal(symbol: str, asset_class: str = "crypto") -> AggregatedSignal | None:
    """Blend sentiment into the latest deterministic signal row for *symbol*.

    Returns:
        AggregatedSignal on success, None when no signal row exists yet.
    """
    conn = get_connection()
    row = conn.execute(
        """
        SELECT id, momentum_score, volatility_score, funding_score, ignition_score
        FROM signals
        WHERE symbol = ? AND asset_class = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (symbol, asset_class),
    ).fetchone()

    if row is None:
        _log.warning("no signal row for %s — skipping aggregation", symbol)
        return None

    signal_id = row["id"]
    mom = float(row["momentum_score"])
    vol = float(row["volatility_score"])
    fund = float(row["funding_score"]) if row["funding_score"] is not None else None
    ign = float(row["ignition_score"]) if row["ignition_score"] is not None else None
    sent = sentiment_signal.compute(symbol)

    composite = _final_composite(mom, vol, fund, sent, asset_class, ign=ign)
    action = _action(composite)

    conn.execute(
        "UPDATE signals SET composite_score = ?, action = ? WHERE id = ?",
        (composite, action, signal_id),
    )
    conn.commit()

    _log.info(
        "aggregated %s: sent=%.3f ign=%s composite=%.3f action=%s",
        symbol,
        sent,
        f"{ign:.3f}" if ign is not None else "n/a",
        composite,
        action,
    )

    return AggregatedSignal(
        symbol=symbol,
        asset_class=asset_class,
        signal_id=signal_id,
        momentum_score=mom,
        volatility_score=vol,
        funding_score=fund,
        sentiment_score=sent,
        composite_score=composite,
        action=action,
        ignition_score=ign,
    )


def aggregate_once(
    crypto_symbols: list[str] | None = None,
    stock_symbols: list[str] | None = None,
) -> list[AggregatedSignal]:
    """Aggregate signals for all configured symbols.

    Args:
        crypto_symbols: Override; defaults to config.CRYPTO_SYMBOLS.
        stock_symbols:  Override; defaults to config.STOCK_SYMBOLS.

    Returns:
        List of AggregatedSignal for every symbol that had a signal row.
    """
    results: list[AggregatedSignal] = []

    for sym in crypto_symbols or config.CRYPTO_SYMBOLS:
        r = aggregate_signal(sym, "crypto")
        if r is not None:
            results.append(r)

    for sym in stock_symbols or config.STOCK_SYMBOLS:
        r = aggregate_signal(sym, "stocks")
        if r is not None:
            results.append(r)

    return results


if __name__ == "__main__":
    for r in aggregate_once():
        print(
            f"{r.symbol:12s}  sent={r.sentiment_score:+.3f}  "
            f"composite={r.composite_score:+.3f}  {r.action}"
        )
