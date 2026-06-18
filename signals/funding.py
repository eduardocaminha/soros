"""Funding rate signal (crypto perpetuals only).

Interprets the perpetual funding rate as a contrarian sentiment indicator:
  - High positive funding → too many longs → bearish contrarian signal
  - High negative funding → too many shorts → bullish contrarian signal

score = -tanh(funding_rate / _SCALE)

_SCALE = 0.0005 (0.05 %) is a typical "elevated" 8-hour funding rate on Binance.
At that rate tanh(1) ≈ 0.76, so extreme rates (> 0.1 %) saturate near ±1.

Returns 0.0 when *funding_rate* is None (stocks or missing data).
"""

from __future__ import annotations

import math


_SCALE = 0.0005  # 0.05 % — normalises typical Binance 8h funding rates


def compute(funding_rate: float | None) -> float:
    """Return funding-rate score.

    Args:
        funding_rate: Latest perpetual funding rate as a decimal (e.g. 0.0001
                      for 0.01 %). Pass ``None`` for stocks or when unavailable.

    Returns:
        Float in [-1, 1]; 0.0 when *funding_rate* is None.
    """
    if funding_rate is None or math.isnan(funding_rate):
        return 0.0

    return float(-math.tanh(funding_rate / _SCALE))
