"""
Dataset builder. Drops look-ahead columns and label-derived columns from X,
restricts candidates to regime-OK bars with a valid label.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# Columns that must NEVER enter X (leakage or non-feature data)
FEATURE_BLOCKLIST = {
    "Date",
    "Open", "High", "Low", "Close", "Volume",       # raw OHLCV (derived features are fine)
    "index",
    "pivot", "pivot_confirmed_at",
    "swing_low", "swing_high",
    "label", "barrier_touched", "forward_return", "forward_bars",
}


def build_dataset(
    df: pd.DataFrame,
    extra_block: List[str] | None = None,
    require_regime: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Returns:
        X       : feature matrix (numeric only)
        y       : binary label
        idx     : positional index back into df for each retained row
    """
    if "label" not in df.columns:
        raise ValueError("call triple_barrier_labels first")

    mask = df["label"].notna()
    if require_regime and "regime_ok" in df.columns:
        mask &= df["regime_ok"] == 1

    sub = df.loc[mask].copy()
    block = set(FEATURE_BLOCKLIST)
    if extra_block:
        block.update(extra_block)
    # Categorical columns we don't want directly fed to xgboost
    block.update(c for c in sub.columns if c == "nearest_fib_level")

    feature_cols = [
        c for c in sub.columns
        if c not in block and pd.api.types.is_numeric_dtype(sub[c])
    ]
    X = sub[feature_cols].astype(float)
    y = sub["label"].astype(int)
    idx = sub.index
    return X, y, idx
