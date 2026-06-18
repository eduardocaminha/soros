"""Volatility breakout signal using Bollinger Bands.

Maps the current close price onto the Bollinger Band range:
  score = (close - lower_band) / (upper_band - lower_band) * 2 - 1

This linearly maps:
  close == lower_band  → -1  (price at lower band, bearish breakout risk)
  close == middle_band →  0  (price at mean, neutral)
  close == upper_band  → +1  (price at upper band, bullish breakout)

Prices beyond the bands are clamped to [-1, 1].

A flat band (std ≈ 0) returns 0.0.
"""

from __future__ import annotations

import math

import pandas as pd

_WINDOW = 20   # look-back period for mean + std
_WIDTH = 2.0   # band multiplier (standard Bollinger default)


def compute(closes: pd.Series) -> float:
    """Return volatility-breakout score for the price series *closes*.

    Args:
        closes: Time-ordered close prices (oldest first). Needs at least
                ``_WINDOW`` rows to produce a non-zero score.

    Returns:
        Float in [-1, 1].
    """
    if len(closes) < _WINDOW:
        return 0.0

    window = closes.iloc[-_WINDOW:]
    mean = float(window.mean())
    std = float(window.std(ddof=1))

    if std == 0.0 or math.isnan(std) or math.isnan(mean):
        return 0.0

    close = float(closes.iloc[-1])
    lower = mean - _WIDTH * std
    upper = mean + _WIDTH * std
    band_range = upper - lower  # == 2 * _WIDTH * std, always > 0

    raw = (close - lower) / band_range * 2.0 - 1.0
    return max(-1.0, min(1.0, raw))
