"""
Per-ticker liquidity / quality stats used to drive the universe filter.
Computed once over a corpus so the sweep does not recompute on every fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class TickerStats:
    ticker: str
    n_bars: int
    median_close: float
    median_dollar_volume: float
    median_atr_pct: float
    first_date: str | None
    last_date: str | None


def _median_atr_pct(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < n + 1:
        return float("nan")
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    return float((atr / df["Close"]).median())


def compute_ticker_stats(corpus: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tk, df in corpus.items():
        if df is None or df.empty:
            continue
        close = df["Close"]
        vol = df["Volume"]
        date_col = df["Date"] if "Date" in df.columns else None
        rows.append(TickerStats(
            ticker=tk,
            n_bars=int(len(df)),
            median_close=float(close.median()),
            median_dollar_volume=float((close * vol).median()),
            median_atr_pct=_median_atr_pct(df),
            first_date=str(date_col.iloc[0]) if date_col is not None else None,
            last_date=str(date_col.iloc[-1]) if date_col is not None else None,
        ).__dict__)
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
