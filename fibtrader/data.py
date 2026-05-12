"""Minimal OHLCV loader. CSV with columns: Date, Open, High, Low, Close, Volume."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    needed = ["open", "high", "low", "close", "volume"]
    for n in needed:
        if n not in cols:
            raise ValueError(f"missing column '{n}' in {path}")
    df = df.rename(columns={cols[n]: n.capitalize() for n in needed})
    if "date" in cols:
        df = df.rename(columns={cols["date"]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df
