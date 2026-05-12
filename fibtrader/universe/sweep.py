"""
Universe sensitivity sweep.

The goal is *not* to find a single best filter — that is a meta-overfitting
trap. Instead, we vary one filter dimension at a time, hold the others at
sensible defaults, and report how strategy performance and universe size
change. The output is meant to be eyeballed (plot or table) so a human can
pick a sweet spot that is also economically interpretable.

This module deliberately avoids automatic grid optimization across all
dimensions simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtest import backtest_fib_confluence
from ..config import BotConfig
from ..features import add_features
from ..fib import attach_recent_swing_and_fib
from ..swing import zigzag_confirmed
from .stats import compute_ticker_stats


DEFAULT_DIMENSIONS: Dict[str, List[float]] = {
    "min_bars":            [252, 504, 756, 1008, 1260],
    "min_dollar_volume":   [0, 5e5, 1e6, 5e6, 2e7, 1e8],
    "min_close":           [0, 1, 3, 5, 10, 30],
    "max_atr_pct":         [0.20, 0.10, 0.05, 0.03, 0.02],
    "min_atr_pct":         [0.0, 0.005, 0.01, 0.015],
}


@dataclass
class FilterSpec:
    min_bars: int = 252
    min_dollar_volume: float = 1e6
    min_close: float = 1.0
    min_atr_pct: float = 0.005
    max_atr_pct: float = 0.10

    def apply(self, stats: pd.DataFrame) -> pd.Index:
        mask = (
            (stats["n_bars"] >= self.min_bars)
            & (stats["median_dollar_volume"] >= self.min_dollar_volume)
            & (stats["median_close"] >= self.min_close)
            & (stats["median_atr_pct"] >= self.min_atr_pct)
            & (stats["median_atr_pct"] <= self.max_atr_pct)
        )
        return stats.index[mask]


def _backtest_one(args) -> Optional[Dict[str, float]]:
    ticker, df, cfg = args
    try:
        sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
        sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
        feat = add_features(sw, cfg)
        trades, stats = backtest_fib_confluence(feat, cfg)
        if stats is None:
            return None
        stats = dict(stats)
        stats["ticker"] = ticker
        return stats
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def backtest_corpus(
    corpus: Dict[str, pd.DataFrame],
    cfg: BotConfig,
    tickers: Optional[Iterable[str]] = None,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Run the backtest on every ticker; returns a DataFrame of per-ticker stats."""
    pool_input = [
        (tk, corpus[tk], cfg)
        for tk in (tickers if tickers is not None else corpus.keys())
        if tk in corpus
    ]
    if n_workers <= 1:
        results = [_backtest_one(a) for a in pool_input]
    else:
        with Pool(n_workers) as p:
            results = p.map(_backtest_one, pool_input)
    rows = [r for r in results if r is not None and "error" not in r]
    return pd.DataFrame(rows)


def _aggregate(per_ticker: pd.DataFrame, metric: str) -> Dict[str, float]:
    if per_ticker.empty or metric not in per_ticker.columns:
        return {
            "n_tickers": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "share_positive": float("nan"),
        }
    s = per_ticker[metric].dropna()
    if s.empty:
        return {"n_tickers": 0, "mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "share_positive": float("nan")}
    return {
        "n_tickers": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "share_positive": float((s > 0).mean()),
    }


def sensitivity_sweep(
    corpus: Dict[str, pd.DataFrame],
    cfg: BotConfig,
    base_filter: Optional[FilterSpec] = None,
    dimensions: Optional[Dict[str, List[float]]] = None,
    metric: str = "sharpe_like",
    per_ticker: Optional[pd.DataFrame] = None,
    ticker_stats: Optional[pd.DataFrame] = None,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Sweep one dimension at a time. Returns long-format rows:
        dimension, threshold, n_tickers, mean, median, p25, p75, share_positive

    Caller can pre-supply per_ticker (avoids re-running backtests) and
    ticker_stats (avoids recomputing liquidity stats).
    """
    base_filter = base_filter or FilterSpec()
    dims = dimensions if dimensions is not None else DEFAULT_DIMENSIONS

    if ticker_stats is None:
        ticker_stats = compute_ticker_stats(corpus)
    if per_ticker is None:
        per_ticker = backtest_corpus(corpus, cfg, n_workers=n_workers)
    if per_ticker.empty:
        return pd.DataFrame()
    indexed = per_ticker.set_index("ticker") if "ticker" in per_ticker.columns else per_ticker

    out_rows = []
    for dim, thresholds in dims.items():
        if dim not in {"min_bars", "min_dollar_volume", "min_close",
                       "min_atr_pct", "max_atr_pct"}:
            continue
        for thr in thresholds:
            spec = FilterSpec(**{**base_filter.__dict__, dim: thr})
            kept = spec.apply(ticker_stats)
            kept_in_results = indexed.index.intersection(kept)
            sub = indexed.loc[kept_in_results]
            agg = _aggregate(sub.reset_index(), metric)
            out_rows.append({
                "dimension": dim,
                "threshold": thr,
                **agg,
            })
    return pd.DataFrame(out_rows)
