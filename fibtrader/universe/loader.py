"""Corpus loader: either a directory of {ticker}.csv files or a long-format CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..data import load_ohlcv_csv


def load_corpus_dir(directory: str | Path, max_tickers: int | None = None) -> Dict[str, pd.DataFrame]:
    d = Path(directory)
    files = sorted(d.glob("*.csv"))
    if max_tickers is not None:
        files = files[:max_tickers]
    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        try:
            df = load_ohlcv_csv(f)
            out[f.stem] = df
        except Exception:
            continue
    return out


def load_corpus_long(path: str | Path, ticker_col: str = "Ticker") -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if ticker_col.lower() not in cols:
        raise ValueError(f"long-format CSV must include '{ticker_col}' column")
    tk_col = cols[ticker_col.lower()]
    rename_targets = {n: n.capitalize() for n in ["open", "high", "low", "close", "volume"]}
    for n in rename_targets:
        if n not in cols:
            raise ValueError(f"long-format CSV missing column '{n}'")
    df = df.rename(columns={cols[n]: rename_targets[n] for n in rename_targets})
    if "date" in cols:
        df = df.rename(columns={cols["date"]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    out: Dict[str, pd.DataFrame] = {}
    for tk, sub in df.groupby(tk_col):
        sub = sub.sort_values("Date" if "Date" in sub.columns else sub.index.name or sub.columns[0])
        out[str(tk)] = sub.reset_index(drop=True)
    return out
