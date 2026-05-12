"""
Massive.io alt-data adapter.

Massive aggregates consumer panel signals (kiosk taps, app usage, etc). We do
not call their API directly here; instead we accept a CSV the user exports
from their dashboard and merge it as additional features.

Expected schema (long format, recommended):
    Date,Ticker,metric,value

Or wide format:
    Date,Ticker,<feature_1>,<feature_2>,...

`merge_altdata` performs a backward as-of merge so a value reported on day D is
visible from day D onward (avoiding look-ahead). It also lags by `release_lag`
business days to model real-world publication delay; default 1 is conservative.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MassiveCSVAdapter:
    path: str | Path
    ticker: Optional[str] = None             # filter to one ticker if file contains many
    long_format: bool = True
    metric_col: str = "metric"
    value_col: str = "value"
    date_col: str = "Date"
    ticker_col: str = "Ticker"
    release_lag_bdays: int = 1

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        if self.date_col not in df.columns:
            raise ValueError(f"alt-data CSV missing column '{self.date_col}'")
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if self.ticker is not None and self.ticker_col in df.columns:
            df = df[df[self.ticker_col] == self.ticker]
        if self.long_format:
            if not {self.metric_col, self.value_col}.issubset(df.columns):
                raise ValueError("long_format=True requires metric/value columns")
            df = df.pivot_table(
                index=self.date_col,
                columns=self.metric_col,
                values=self.value_col,
                aggfunc="last",
            ).reset_index()
            df.columns.name = None
        # Lag for publication delay
        if self.release_lag_bdays > 0:
            df[self.date_col] = df[self.date_col] + pd.tseries.offsets.BusinessDay(
                self.release_lag_bdays
            )
        # Prefix to make merged columns identifiable
        rename = {c: f"alt_{c}" for c in df.columns if c != self.date_col}
        df = df.rename(columns=rename)
        return df.sort_values(self.date_col).reset_index(drop=True)


def merge_altdata(price_df: pd.DataFrame, alt: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if date_col not in price_df.columns:
        raise ValueError("price_df must have a Date column for as-of merging")
    p = price_df.copy()
    p[date_col] = pd.to_datetime(p[date_col])
    p = p.sort_values(date_col).reset_index(drop=True)
    merged = pd.merge_asof(p, alt, on=date_col, direction="backward")
    return merged
