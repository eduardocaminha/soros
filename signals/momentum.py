"""Momentum signal based on EMA crossover.

Returns a score in [-1, 1]:
  +1 = strong upward momentum (fast EMA well above slow EMA)
  -1 = strong downward momentum (fast EMA well below slow EMA)
   0 = no momentum (EMAs aligned)

The relative EMA spread is passed through tanh to bound the output and
reduce sensitivity to outliers.
"""

from __future__ import annotations

import math

import pandas as pd

# EMA periods (standard MACD parameters)
_FAST = 12
_SLOW = 26

# Scale factor inside tanh: a 0.5% EMA spread → score ≈ 0.46
# Chosen so typical crypto moves produce meaningful (non-saturated) scores.
_SCALE = 200.0


def compute(closes: pd.Series) -> float:
    """Return momentum score for the price series *closes*.

    Args:
        closes: Time-ordered close prices (oldest first). Needs at least
                ``_SLOW`` rows to produce a non-zero score.

    Returns:
        Float in [-1, 1].
    """
    if len(closes) < _SLOW:
        return 0.0

    fast_ema = closes.ewm(span=_FAST, adjust=False).mean().iloc[-1]
    slow_ema = closes.ewm(span=_SLOW, adjust=False).mean().iloc[-1]

    if slow_ema == 0.0 or math.isnan(slow_ema) or math.isnan(fast_ema):
        return 0.0

    spread = (fast_ema - slow_ema) / slow_ema
    return float(math.tanh(spread * _SCALE))
