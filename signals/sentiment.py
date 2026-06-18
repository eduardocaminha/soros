"""Sentiment signal — 4th signal, reads latest row from sentiment_signals.

Returns the most recent score for a symbol if it is fresh enough
(age <= config.SENTIMENT_MAX_AGE_SECONDS).  Returns 0.0 (neutral) when:
  - no row exists for the symbol
  - the latest row is stale
  - config.SENTIMENT_ENABLED is False
"""

from __future__ import annotations

import time

import config
from database.db import get_connection


def compute(symbol: str, *, now: int | None = None) -> float:
    """Return the current sentiment score for *symbol* in [-1, 1].

    Args:
        symbol: Trading symbol (e.g. ``'BTC/USDT'``, ``'AAPL'``).
        now:    Unix-seconds override for the current time (testing only).

    Returns:
        Float in [-1, 1].  0.0 when sentiment is disabled, absent, or stale.
    """
    if not config.SENTIMENT_ENABLED:
        return 0.0

    if now is None:
        now = int(time.time())

    conn = get_connection()
    row = conn.execute(
        """
        SELECT score, ts
        FROM sentiment_signals
        WHERE symbol = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()

    if row is None:
        return 0.0

    score: float = row[0]
    ts: int = row[1]

    if now - ts > config.SENTIMENT_MAX_AGE_SECONDS:
        return 0.0

    return max(-1.0, min(1.0, float(score)))
