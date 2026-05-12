"""
Statistical evaluation: trade-level perf metrics and a permutation test that
swaps Fibonacci ratios for random ratios on the same swings, returning a
two-sided empirical p-value and an effect size.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .backtest import backtest_fib_confluence
from .config import BotConfig
from .features import add_features
from .fib import attach_recent_swing_and_fib


def summarize_trades(trades: pd.DataFrame, cfg: BotConfig) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {"trades": 0}
    rets = trades["ret"]
    eq = (1.0 + rets).cumprod()
    dd = eq / eq.cummax() - 1.0
    sharpe = (rets.mean() / (rets.std(ddof=1) + 1e-12)) * np.sqrt(cfg.annualization_factor)
    return {
        "trades": int(len(trades)),
        "win_rate": float((rets > 0).mean()),
        "avg_ret": float(rets.mean()),
        "sharpe_like": float(sharpe),
        "max_drawdown": float(dd.min()),
        "cum_return": float(eq.iloc[-1] - 1.0),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    if sp == 0:
        return float("nan")
    return (a.mean() - b.mean()) / sp


def permutation_test_random_ratios(
    df_with_swings: pd.DataFrame,
    cfg: BotConfig,
    metric: str = "avg_ret",
) -> Dict[str, float]:
    """
    Null hypothesis: random 5-ratio sets within [random_ratio_low, random_ratio_high]
    yield the same `metric` as the canonical Fib ratios.

    df_with_swings must already have pivot/pivot_confirmed_at/swing columns from
    zigzag_confirmed; this function rebuilds the fib levels for each random draw.
    """
    rng = np.random.default_rng(cfg.random_seed)

    # Baseline: canonical Fibonacci ratios
    baseline_df = attach_recent_swing_and_fib(df_with_swings, ratios=cfg.fib_ratios)
    baseline_feat = add_features(baseline_df, cfg)
    _, baseline_stats = backtest_fib_confluence(baseline_feat, cfg)
    if baseline_stats is None:
        return {"p_value": float("nan"), "baseline": float("nan"), "n_perms": 0}
    baseline_metric = baseline_stats[metric]

    perm_metrics = []
    for _ in range(cfg.n_permutations):
        ratios = sorted(
            set(
                np.round(
                    rng.uniform(cfg.random_ratio_low, cfg.random_ratio_high, size=len(cfg.fib_ratios) * 2),
                    3,
                )
            )
        )[: len(cfg.fib_ratios)]
        if len(ratios) < len(cfg.fib_ratios):
            continue
        rdf = attach_recent_swing_and_fib(df_with_swings, ratios=tuple(ratios))
        rfeat = add_features(rdf, cfg)
        _, rstats = backtest_fib_confluence(rfeat, cfg)
        if rstats is None:
            perm_metrics.append(0.0)
        else:
            perm_metrics.append(rstats[metric])

    perm_arr = np.array(perm_metrics, dtype=float)
    # Two-sided empirical p-value
    p_value = float((np.abs(perm_arr - perm_arr.mean()) >= abs(baseline_metric - perm_arr.mean())).mean())
    effect = _cohens_d(np.array([baseline_metric]), perm_arr) if len(perm_arr) > 1 else float("nan")
    return {
        "metric": metric,
        "baseline": float(baseline_metric),
        "perm_mean": float(perm_arr.mean()) if len(perm_arr) else float("nan"),
        "perm_std": float(perm_arr.std(ddof=1)) if len(perm_arr) > 1 else float("nan"),
        "p_value": p_value,
        "effect_size_d": float(effect),
        "n_perms": int(len(perm_arr)),
    }
