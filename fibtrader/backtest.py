"""
Event-driven long-only backtester.

Conservative assumptions:
  - Signal evaluated on bar i; entry executed at Open of bar i+1.
  - Costs (fee + slippage) applied to both entry and exit.
  - If stop and target both hit within the same bar, STOP wins (worst case).
  - One position at a time; no pyramiding.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import BotConfig


def backtest_fib_confluence(
    df: pd.DataFrame,
    cfg: BotConfig,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    data = df.copy().reset_index(drop=True)

    required = {"Open", "High", "Low", "Close", "atr", "confluence_count", "fib_near", "trend_up", "regime_ok"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns for backtest: {missing}")

    cost = (cfg.fee_bps + cfg.slippage_bps) / 1e4

    trades = []
    i = 0
    n = len(data)

    while i < n - 1:
        row = data.iloc[i]

        if (
            row["fib_near"] == 1
            and row["confluence_count"] >= cfg.min_confluence
            and row["trend_up"] == 1
            and row["regime_ok"] == 1
            and not np.isnan(row["atr"])
        ):
            entry_idx = i + 1
            if entry_idx >= n:
                break
            entry_open = data.iloc[entry_idx]["Open"]
            if np.isnan(entry_open) or np.isnan(row["atr"]):
                i += 1
                continue

            entry_price = entry_open * (1.0 + cost)
            atr_now = max(row["atr"], 1e-8)
            stop = entry_price - cfg.stop_atr_mult * atr_now
            take = entry_price + cfg.take_atr_mult * atr_now

            exit_idx = None
            exit_price = None
            for j in range(entry_idx + 1, min(entry_idx + cfg.hold_bars + 1, n)):
                low_j = data.iloc[j]["Low"]
                high_j = data.iloc[j]["High"]
                if low_j <= stop:
                    exit_idx = j
                    exit_price = stop * (1.0 - cost)
                    exit_reason = "stop"
                    break
                if high_j >= take:
                    exit_idx = j
                    exit_price = take * (1.0 - cost)
                    exit_reason = "take"
                    break

            if exit_idx is None:
                exit_idx = min(entry_idx + cfg.hold_bars, n - 1)
                exit_price = data.iloc[exit_idx]["Close"] * (1.0 - cost)
                exit_reason = "time"

            ret = (exit_price - entry_price) / entry_price
            trades.append(
                {
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret": ret,
                    "bars_held": exit_idx - entry_idx,
                    "confluence": int(row["confluence_count"]),
                    "exit_reason": exit_reason,
                    "nearest_fib": row.get("nearest_fib_level", None),
                }
            )
            i = exit_idx + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, None

    eq = (1.0 + trades_df["ret"]).cumprod() * cfg.initial_equity
    running_max = eq.cummax()
    dd = eq / running_max - 1.0

    rets = trades_df["ret"]
    sharpe = (rets.mean() / (rets.std(ddof=1) + 1e-12)) * np.sqrt(cfg.annualization_factor)
    downside = rets[rets < 0]
    sortino = (
        rets.mean() / (downside.std(ddof=1) + 1e-12) * np.sqrt(cfg.annualization_factor)
        if len(downside) > 1
        else float("nan")
    )
    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    profit_factor = gains / losses if losses > 0 else float("inf")

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float((rets > 0).mean()),
        "avg_ret": float(rets.mean()),
        "median_ret": float(rets.median()),
        "sharpe_like": float(sharpe),
        "sortino_like": float(sortino),
        "max_drawdown": float(dd.min()),
        "cum_return": float(eq.iloc[-1] / cfg.initial_equity - 1.0),
        "profit_factor": float(profit_factor),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
    }
    return trades_df, stats


def walk_forward_backtest(
    df: pd.DataFrame,
    cfg: BotConfig,
    n_splits: int = 5,
):
    """Yield (split_id, train_slice, test_slice, trades, stats) per fold."""
    n = len(df)
    fold = n // (n_splits + 1)
    for k in range(n_splits):
        train = df.iloc[: fold * (k + 1)]
        test = df.iloc[fold * (k + 1) : fold * (k + 2)]
        if len(test) == 0:
            continue
        trades, stats = backtest_fib_confluence(test, cfg)
        yield k, train, test, trades, stats
