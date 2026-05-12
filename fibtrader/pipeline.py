"""
공통 feature 파이프라인.

zigzag_confirmed → attach_recent_swing_and_fib → add_features 시퀀스가
백테스트·풀드 학습·sweep·CLI 등 5곳에서 반복되어 한 곳으로 모음.
"""
from __future__ import annotations

import pandas as pd

from .config import BotConfig
from .features import add_features
from .fib import attach_recent_swing_and_fib
from .swing import zigzag_confirmed


def prepare_features(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    """OHLCV → swing → fib → feature 까지 한 번에."""
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    return add_features(sw, cfg)
