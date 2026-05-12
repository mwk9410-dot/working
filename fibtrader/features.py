"""
Feature engineering on top of OHLCV + confirmed swing/fib.

Produces a confluence-count column used by the backtester, plus ML-ready
columns (distances, regime flags) for downstream model training.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import BotConfig
from .indicators import atr, bollinger, macd, rsi, vol_zscore


def add_features(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    out = df.copy()

    out["ma20"] = out["Close"].rolling(20).mean()
    out["ma50"] = out["Close"].rolling(50).mean()
    out["ma200"] = out["Close"].rolling(200).mean()

    out["rsi"] = rsi(out["Close"], cfg.rsi_len)
    macd_line, signal_line, hist = macd(
        out["Close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    out["macd"] = macd_line
    out["macd_sig"] = signal_line
    out["macd_hist"] = hist

    bb_mid, bb_up, bb_low = bollinger(out["Close"], cfg.bb_len, cfg.bb_k)
    out["bb_mid"] = bb_mid
    out["bb_up"] = bb_up
    out["bb_low"] = bb_low

    out["atr"] = atr(out, cfg.atr_len)
    out["atr_pct"] = out["atr"] / out["Close"]
    out["vol_z"] = vol_zscore(out["Volume"], cfg.vol_lookback)

    fib_cols = [c for c in out.columns if c.startswith("fib_")]
    if not fib_cols:
        raise ValueError("expected fib_* columns; call attach_recent_swing_and_fib first")

    # Distance to each fib level (NaN if level not yet defined)
    for c in fib_cols:
        out[f"dist_{c}"] = (out["Close"] - out[c]).abs() / out["Close"]

    dist_cols = [f"dist_{c}" for c in fib_cols]
    dist_matrix = out[dist_cols]
    out["nearest_fib_dist"] = dist_matrix.min(axis=1)
    any_valid = dist_matrix.notna().any(axis=1)
    nearest = pd.Series(index=out.index, dtype=object)
    if any_valid.any():
        nearest.loc[any_valid] = dist_matrix.loc[any_valid].idxmin(axis=1)
    out["nearest_fib_level"] = nearest

    # Retracement depth: how far the current close has retraced inside the leg
    leg_size = out["swing_high"] - out["swing_low"]
    out["retracement_depth"] = np.where(
        leg_size > 0,
        (out["swing_high"] - out["Close"]) / leg_size,
        np.nan,
    )

    # Confluence components
    out["trend_up"] = ((out["ma20"] > out["ma50"]) & (out["ma50"] > out["ma200"])).astype(int)
    out["rsi_reversal"] = ((out["rsi"].shift(1) < cfg.rsi_oversold) & (out["rsi"].diff() > 0)).astype(int)
    out["macd_confirm"] = ((out["macd_hist"] > 0) & (out["macd_hist"].diff() > 0)).astype(int)
    out["bb_confirm"] = (out["Close"] < out["bb_low"] * 1.01).astype(int)
    out["vol_confirm"] = (out["vol_z"] > cfg.vol_z_threshold).astype(int)
    out["fib_near"] = (out["nearest_fib_dist"] < cfg.fib_proximity_pct).astype(int)

    out["regime_ok"] = (
        (out["atr_pct"] >= cfg.atr_pct_min) & (out["atr_pct"] <= cfg.atr_pct_max)
    ).astype(int)

    out["confluence_count"] = (
        out["fib_near"]
        + out["trend_up"]
        + out["rsi_reversal"]
        + out["macd_confirm"]
        + out["bb_confirm"]
        + out["vol_confirm"]
    )
    return out
