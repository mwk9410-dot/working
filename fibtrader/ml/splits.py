"""
Embargoed walk-forward splitter.

For overlapping forward labels (triple-barrier with hold_bars > 1) we must
exclude the `embargo` bars immediately following each test window from the
training set in subsequent folds. Here we use a simpler but valid form: drop
the embargo bars at the BOUNDARY between train and test in every fold.

Use the `gap` to also leave an embargo BEFORE the test (preventing the
training set from including bars whose forward window overlaps the test
start).
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
