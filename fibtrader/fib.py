"""
Fibonacci levels derived from the most recently CONFIRMED up-leg
(swing low -> swing high). Only fills levels from the bar where the pivot is
known to the user, so no look-ahead leaks into features.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

DEFAULT_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786)


def fib_levels_from_leg(
    low_price: float,
    high_price: float,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> Dict[str, float]:
    diff = high_price - low_price
    out = {f"fib_{int(r * 1000):03d}": high_price - r * diff for r in ratios}
    out["ext_1272"] = high_price + 0.272 * diff
    out["ext_1618"] = high_price + 0.618 * diff
    return out


def attach_recent_swing_and_fib(
    df: pd.DataFrame,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> pd.DataFrame:
    """
    df must already have columns: pivot, pivot_confirmed_at, High, Low.
    Produces fib_xxx columns valid only from the bar each up-leg was confirmed.
    """
    out = df.copy()
    n = len(out)
    out["swing_low"] = np.nan
    out["swing_high"] = np.nan
    for r in ratios:
        out[f"fib_{int(r * 1000):03d}"] = np.nan

    pivot = out["pivot"].to_numpy()
    conf_at = out["pivot_confirmed_at"].to_numpy()
    highs = out["High"].to_numpy()
    lows = out["Low"].to_numpy()

    # Pre-sort pivots by their confirmation bar
    pivot_events = []  # (confirmed_at, pivot_bar, type)
    for i in range(n):
        if pivot[i] != 0 and conf_at[i] >= 0:
            pivot_events.append((conf_at[i], i, int(pivot[i])))
    pivot_events.sort()

    last_low = None
    last_low_bar = None
    last_high = None
    last_high_bar = None
    # current active up-leg (low confirmed, then high confirmed AFTER it)
    leg_low = None
    leg_high = None
    leg_high_confirmed_at = None

    fib_arrays = {f"fib_{int(r * 1000):03d}": np.full(n, np.nan) for r in ratios}
    swing_low_arr = np.full(n, np.nan)
    swing_high_arr = np.full(n, np.nan)

    ev_iter = iter(pivot_events)
    next_event = next(ev_iter, None)

    for bar in range(n):
        # Apply every event whose confirmation index <= bar
        while next_event is not None and next_event[0] <= bar:
            conf_bar, piv_bar, piv_type = next_event
            if piv_type == 1:  # pivot low
                last_low = lows[piv_bar]
                last_low_bar = piv_bar
                # Up-leg requires a low followed by a high AFTER it
            elif piv_type == -1:  # pivot high
                last_high = highs[piv_bar]
                last_high_bar = piv_bar
                if (
                    last_low is not None
                    and last_low_bar is not None
                    and last_high_bar > last_low_bar
                    and last_high > last_low
                ):
                    leg_low = last_low
                    leg_high = last_high
                    leg_high_confirmed_at = conf_bar
            next_event = next(ev_iter, None)

        if leg_low is not None and leg_high is not None and leg_high_confirmed_at is not None and bar >= leg_high_confirmed_at:
            swing_low_arr[bar] = leg_low
            swing_high_arr[bar] = leg_high
            levels = fib_levels_from_leg(leg_low, leg_high, ratios)
            for r in ratios:
                key = f"fib_{int(r * 1000):03d}"
                fib_arrays[key][bar] = levels[key]

    out["swing_low"] = swing_low_arr
    out["swing_high"] = swing_high_arr
    for key, arr in fib_arrays.items():
        out[key] = arr
    return out
