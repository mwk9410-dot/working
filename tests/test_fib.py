import numpy as np

from fibtrader.fib import attach_recent_swing_and_fib, fib_levels_from_leg
from fibtrader.swing import zigzag_confirmed


def test_fib_levels_basic():
    levels = fib_levels_from_leg(100.0, 200.0)
    assert np.isclose(levels["fib_500"], 150.0)
    assert np.isclose(levels["fib_618"], 200.0 - 0.618 * 100.0)
    assert np.isclose(levels["fib_236"], 200.0 - 0.236 * 100.0)


def test_fib_no_lookahead(synth_df):
    sw = zigzag_confirmed(synth_df, pct=0.05, confirm_bars=2)
    sw_fib = attach_recent_swing_and_fib(sw)
    # No fib value should be set before the FIRST up-leg has been confirmed.
    pivot_lows = sw[sw["pivot"] == 1]
    pivot_highs = sw[sw["pivot"] == -1]
    if pivot_lows.empty or pivot_highs.empty:
        return
    # Earliest possible up-leg confirmation bar
    candidates = []
    for h_idx, h_row in pivot_highs.iterrows():
        prior_lows = pivot_lows[pivot_lows.index < h_idx]
        if not prior_lows.empty:
            candidates.append(int(h_row["pivot_confirmed_at"]))
    assert candidates, "test data should produce at least one up-leg"
    earliest = min(candidates)
    pre = sw_fib.iloc[:earliest]
    assert pre[["fib_236", "fib_382", "fib_500", "fib_618", "fib_786"]].isna().all().all()
