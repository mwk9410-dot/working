"""
Confirmed-pivot ZigZag.

Look-ahead control: a pivot is only marked at its bar AFTER price has moved at
least `pct` against the candidate AND `confirm_bars` additional bars have passed
since the candidate was set. Downstream code must use `pivot_confirmed_at` (the
bar index where the pivot becomes usable), not the bar of the pivot itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zigzag_confirmed(df: pd.DataFrame, pct: float = 0.05, confirm_bars: int = 2) -> pd.DataFrame:
    """
    Returns a copy of df with columns:
      pivot                : 1 = pivot low, -1 = pivot high, 0 = none (placed at pivot bar)
      pivot_confirmed_at   : index where the pivot is first known (>= pivot bar)
    """
    out = df.copy().reset_index(drop=False)
    close = out["Close"].to_numpy()
    n = len(close)

    pivot = np.zeros(n, dtype=int)
    confirmed_at = np.full(n, -1, dtype=int)

    last_pivot_idx = 0
    last_pivot_price = close[0]
    trend = 0  # 1 up-leg, -1 down-leg, 0 unknown
    candidate_high_idx = 0
    candidate_low_idx = 0

    for i in range(1, n):
        price = close[i]

        if trend == 0:
            up_move = (price - last_pivot_price) / last_pivot_price
            dn_move = (last_pivot_price - price) / last_pivot_price
            if up_move >= pct:
                pivot[last_pivot_idx] = 1
                confirmed_at[last_pivot_idx] = i + confirm_bars
                trend = 1
                candidate_high_idx = i
            elif dn_move >= pct:
                pivot[last_pivot_idx] = -1
                confirmed_at[last_pivot_idx] = i + confirm_bars
                trend = -1
                candidate_low_idx = i

        elif trend == 1:
            if close[i] >= close[candidate_high_idx]:
                candidate_high_idx = i
            reversal = (close[candidate_high_idx] - price) / close[candidate_high_idx]
            if reversal >= pct:
                pivot[candidate_high_idx] = -1
                confirmed_at[candidate_high_idx] = i + confirm_bars
                last_pivot_idx = candidate_high_idx
                last_pivot_price = close[candidate_high_idx]
                trend = -1
                candidate_low_idx = i

        else:  # trend == -1
            if close[i] <= close[candidate_low_idx]:
                candidate_low_idx = i
            reversal = (price - close[candidate_low_idx]) / close[candidate_low_idx]
            if reversal >= pct:
                pivot[candidate_low_idx] = 1
                confirmed_at[candidate_low_idx] = i + confirm_bars
                last_pivot_idx = candidate_low_idx
                last_pivot_price = close[candidate_low_idx]
                trend = 1
                candidate_high_idx = i

    out["pivot"] = pivot
    out["pivot_confirmed_at"] = confirmed_at
    return out
