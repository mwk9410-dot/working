from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BotConfig:
    # Swing detection
    zigzag_pct: float = 0.05          # reversal threshold for ZigZag (e.g. 5%)
    confirm_bars: int = 2              # extra bars before a pivot is considered confirmed

    # Fibonacci
    fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.500, 0.618, 0.786)
    fib_proximity_pct: float = 0.005   # within 0.5% of a fib level counts as "near"

    # Indicators
    rsi_len: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_len: int = 20
    bb_k: float = 2.0
    atr_len: int = 14
    vol_lookback: int = 50
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    vol_z_threshold: float = 1.0

    # Regime filter (ATR%)
    atr_pct_min: float = 0.005         # skip dead markets
    atr_pct_max: float = 0.10          # skip pathological volatility

    # Entry / exit
    min_confluence: int = 3            # require >= 3 confirmations including fib_near
    hold_bars: int = 20
    stop_atr_mult: float = 1.5
    take_atr_mult: float = 2.5

    # Costs (one-way, in basis points)
    fee_bps: float = 5.0
    slippage_bps: float = 5.0

    # Backtest
    annualization_factor: float = 252.0   # daily; override for intraday
    initial_equity: float = 1.0

    # Permutation test
    n_permutations: int = 1000
    random_ratio_low: float = 0.2
    random_ratio_high: float = 0.8
    random_seed: int = 42
