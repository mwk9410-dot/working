"""
Live (or paper) signal generator. Wraps the same pipeline used in backtest so
the offline-vs-online distribution stays identical.

Usage:
    gen = LiveSignalGenerator(cfg)
    gen.warmup(history_df)        # last ~250 bars
    decision = gen.on_new_bar(bar)  # returns Decision or None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import BotConfig
from .features import add_features
from .fib import attach_recent_swing_and_fib
from .swing import zigzag_confirmed


@dataclass
class Decision:
    action: str               # "enter_long" | "hold" | "exit"
    price_hint: float
    stop: float
    take: float
    confluence: int
    nearest_fib: str
    reason: str


class LiveSignalGenerator:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.history: pd.DataFrame = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        self.in_position = False
        self.entry_price: Optional[float] = None
        self.stop: Optional[float] = None
        self.take: Optional[float] = None
        self.bars_held = 0

    def warmup(self, df: pd.DataFrame) -> None:
        self.history = df[["Open", "High", "Low", "Close", "Volume"]].copy().reset_index(drop=True)

    def on_new_bar(self, bar: pd.Series) -> Optional[Decision]:
        self.history = pd.concat([self.history, bar.to_frame().T], ignore_index=True)

        if self.in_position:
            self.bars_held += 1
            low = bar["Low"]
            high = bar["High"]
            if low <= self.stop:
                self._flatten()
                return Decision("exit", self.stop, self.stop, self.take, 0, "", "stop_hit")
            if high >= self.take:
                self._flatten()
                return Decision("exit", self.take, self.stop, self.take, 0, "", "take_hit")
            if self.bars_held >= self.cfg.hold_bars:
                self._flatten()
                return Decision("exit", bar["Close"], self.stop, self.take, 0, "", "time_stop")
            return None

        # Not in position: evaluate entry on the most recent CLOSED bar
        # (warmup must include enough bars for indicators)
        if len(self.history) < 220:
            return None

        sw = zigzag_confirmed(self.history, pct=self.cfg.zigzag_pct, confirm_bars=self.cfg.confirm_bars)
        sw = attach_recent_swing_and_fib(sw, ratios=self.cfg.fib_ratios)
        feat = add_features(sw, self.cfg)
        row = feat.iloc[-1]

        if (
            row["fib_near"] == 1
            and row["confluence_count"] >= self.cfg.min_confluence
            and row["trend_up"] == 1
            and row["regime_ok"] == 1
            and not np.isnan(row["atr"])
        ):
            entry_hint = bar["Close"]
            atr_now = max(row["atr"], 1e-8)
            stop = entry_hint - self.cfg.stop_atr_mult * atr_now
            take = entry_hint + self.cfg.take_atr_mult * atr_now
            self.in_position = True
            self.entry_price = entry_hint
            self.stop = stop
            self.take = take
            self.bars_held = 0
            return Decision(
                action="enter_long",
                price_hint=entry_hint,
                stop=stop,
                take=take,
                confluence=int(row["confluence_count"]),
                nearest_fib=str(row.get("nearest_fib_level", "")),
                reason="fib_confluence",
            )
        return None

    def _flatten(self) -> None:
        self.in_position = False
        self.entry_price = None
        self.stop = None
        self.take = None
        self.bars_held = 0
