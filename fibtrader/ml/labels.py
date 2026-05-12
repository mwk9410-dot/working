"""
Triple-barrier labeling (López de Prado, AFML).

For each candidate bar t we simulate the same execution model used by the
backtester:
  entry  = Open[t+1] * (1 + cost)
  stop   = entry - stop_atr_mult * atr[t]
  take   = entry + take_atr_mult * atr[t]
  vert   = t + 1 + hold_bars

label = 1  if upper (take) hit first
        0  otherwise (stop first, or vertical first with non-positive return)

This matches the backtester's "stop wins ties" rule, so the labels and the
post-training paper backtest agree on what counts as a winning trade.

Returned columns:
    label                  : 0/1 (NaN if no valid forward window)
    barrier_touched        : "take" | "stop" | "time" | NaN
    forward_return         : return at exit (after cost both sides)
    forward_bars           : bars held until exit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BotConfig


def triple_barrier_labels(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "atr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for labeling: {missing}")

    n = len(df)
    cost = (cfg.fee_bps + cfg.slippage_bps) / 1e4

    label = np.full(n, np.nan)
    touched = np.array([None] * n, dtype=object)
    fwd_ret = np.full(n, np.nan)
    fwd_bars = np.full(n, np.nan)

    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    atrs = df["atr"].to_numpy()

    for t in range(n - 1):
        a = atrs[t]
        o_next = opens[t + 1]
        if not np.isfinite(a) or not np.isfinite(o_next) or a <= 0:
            continue

        entry = o_next * (1.0 + cost)
        stop = entry - cfg.stop_atr_mult * a
        take = entry + cfg.take_atr_mult * a

        last = min(t + 1 + cfg.hold_bars, n - 1)
        exit_idx = None
        exit_price = None
        result = None
        for j in range(t + 2, last + 1):
            if lows[j] <= stop:
                exit_idx, exit_price, result = j, stop * (1.0 - cost), "stop"
                break
            if highs[j] >= take:
                exit_idx, exit_price, result = j, take * (1.0 - cost), "take"
                break

        if exit_idx is None:
            exit_idx = last
            exit_price = closes[last] * (1.0 - cost)
            result = "time"

        r = (exit_price - entry) / entry
        label[t] = 1.0 if result == "take" or (result == "time" and r > 0) else 0.0
        touched[t] = result
        fwd_ret[t] = r
        fwd_bars[t] = exit_idx - (t + 1)

    out = df.copy()
    out["label"] = label
    out["barrier_touched"] = touched
    out["forward_return"] = fwd_ret
    out["forward_bars"] = fwd_bars
    return out
