"""Volatility breakout signal using ATR-based channel (Keltner Channel).

Channel = EMA(close, n) +/- k * ATR(n)
Score = position of close relative to the channel, clamped to [-1, 1]:
  close >= upper_band  -> +1  (bullish breakout)
  close <= lower_band  -> -1  (bearish breakout)
  between bands        -> linear interpolation

Returns 0.0 when data is insufficient or ATR is zero.
"""

from __future__ import annotations

import math

import pandas as pd

_WINDOW = 20   # EMA and ATR look-back period
_MULT = 2.0    # ATR multiplier (channel half-width)


def compute(df: pd.DataFrame) -> float:
    """Return volatility-breakout score for the OHLCV data *df*.

    Args:
        df: DataFrame with columns ['high', 'low', 'close'], time-ordered
            oldest first. Needs at least _WINDOW + 1 rows to produce a
            non-zero score (ATR needs one prior close for True Range).

    Returns:
        Float in [-1, 1].
    """
    if len(df) < _WINDOW + 1:
        return 0.0

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)

    prev_close = closes.shift(1)
    tr = pd.concat(
        [highs - lows, (highs - prev_close).abs(), (lows - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = float(tr.rolling(_WINDOW).mean().iloc[-1])
    ema = float(closes.ewm(span=_WINDOW, adjust=False).mean().iloc[-1])
    close = float(closes.iloc[-1])

    if math.isnan(atr) or math.isnan(ema) or atr == 0.0:
        return 0.0

    upper = ema + _MULT * atr
    lower = ema - _MULT * atr
    band_range = upper - lower  # == 2 * _MULT * atr, always > 0

    raw = (close - lower) / band_range * 2.0 - 1.0
    return max(-1.0, min(1.0, raw))
