"""
Embargoed walk-forward splitters.

`EmbargoedWalkForward`: K chunked folds with an embargo gap. Fast, suitable
for first-pass OOS estimates.

`DailyRollingWalkForward`: bar-by-bar rolling validation. For each test point
t in [min_train, n), the training window is [0, t - embargo) and the test
window is a single bar at t (or a small slice if `test_window > 1`). This
matches the "train at D, validate at D+gap" pattern where `gap == embargo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class EmbargoedWalkForward:
    n_splits: int = 5
    embargo: int = 20      # set to hold_bars (or larger)
    min_train: int = 200

    def split(self, n_rows: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if n_rows < self.min_train + self.n_splits * 2:
            raise ValueError("not enough rows for the requested number of splits")

        fold_size = (n_rows - self.min_train) // self.n_splits
        for k in range(self.n_splits):
            test_start = self.min_train + k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n_rows
            train_end = max(0, test_start - self.embargo)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


@dataclass
class DailyRollingWalkForward:
    """
    Bar-by-bar rolling walk-forward.

    Each fold yields:
        train_idx = [0, t - embargo)
        test_idx  = [t, t + test_window)

    Defaults: step=1 (advance one bar per fold), test_window=1 (one bar per fold).
    For daily data with hold_bars=20, set embargo>=20 to prevent label leakage.
    """
    min_train: int = 252
    embargo: int = 20
    step: int = 1
    test_window: int = 1

    def split(self, n_rows: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        t = self.min_train
        while t + self.test_window <= n_rows:
            train_end = max(0, t - self.embargo)
            if train_end <= 0:
                t += self.step
                continue
            yield np.arange(0, train_end), np.arange(t, t + self.test_window)
            t += self.step

