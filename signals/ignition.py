"""Ignition signal: volume z-score + price rate-of-change (crypto only).

Detects sudden volume surges combined with rapid price movement — the
characteristic pattern of an early breakout or gem move.

score = 0.5 * tanh(volume_z / _Z_SCALE) + 0.5 * tanh(roc / _ROC_SCALE)

Returns 0.0 when there is insufficient data for either component.
"""

from __future__ import annotations

import math

import pandas as pd

# Reference window for rolling volume mean/std (prior bars, excluding latest)
_VOL_WINDOW = 20

# Lookback for price rate-of-change (fast-mover detection)
_ROC_WINDOW = 5

# Volume z-score of 1.0 → tanh(1.0) ≈ 0.76; typical spike produces a
# meaningful non-saturated score without hitting extremes for normal variance.
_Z_SCALE = 1.0

# 5 % price move → tanh(1.0) ≈ 0.76; aligns with GEM_ROC_MIN_PCT default.
_ROC_SCALE = 0.05


def compute(df: pd.DataFrame) -> float:
    """Return ignition score for *df*.

    Args:
        df: DataFrame with columns ``close`` and ``volume``, ordered
            oldest-first. Needs at least ``_VOL_WINDOW + 1`` rows.

    Returns:
        Float in [-1, 1]; 0.0 when data is insufficient or degenerate.
    """
    if len(df) < _VOL_WINDOW + 1:
        return 0.0

    # --- Volume z-score (latest bar vs prior _VOL_WINDOW bars) ---
    vol = df["volume"]
    ref_vol = vol.iloc[-_VOL_WINDOW - 1 : -1]  # prior _VOL_WINDOW bars
    latest_vol = float(vol.iloc[-1])
    vol_mean = float(ref_vol.mean())
    vol_std = float(ref_vol.std(ddof=1))

    if vol_std == 0.0 or math.isnan(vol_std):
        vol_sig = 0.0
    else:
        z = (latest_vol - vol_mean) / vol_std
        vol_sig = math.tanh(z / _Z_SCALE)

    # --- Price rate of change ---
    closes = df["close"]
    ref_close = float(closes.iloc[-(_ROC_WINDOW + 1)])
    cur_close = float(closes.iloc[-1])

    if ref_close == 0.0 or math.isnan(ref_close) or math.isnan(cur_close):
        roc_sig = 0.0
    else:
        roc = (cur_close - ref_close) / ref_close
        roc_sig = math.tanh(roc / _ROC_SCALE)

    return float((vol_sig + roc_sig) / 2.0)
