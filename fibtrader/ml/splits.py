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
    """
    K-fold walk-forward 검증기.

    max_train_size 가 None 이면 누적식(expanding) — 학습 구간이 fold 진행될수록
    계속 커짐. 정수면 고정식(sliding) — 학습 구간 크기를 max_train_size 봉으로
    잘라서 항상 같은 크기 유지. 6500종목 같은 대규모 풀드 학습에는 sliding 추천.
    """
    n_splits: int = 5
    embargo: int = 20      # hold_bars 이상 권장
    min_train: int = 200
    max_train_size: int | None = None  # None=누적식, 정수=고정 크기 sliding

    def split(self, n_rows: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if n_rows < self.min_train + self.n_splits * 2:
            raise ValueError("fold 수에 비해 데이터가 너무 적음")

        fold_size = (n_rows - self.min_train) // self.n_splits
        for k in range(self.n_splits):
            test_start = self.min_train + k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n_rows
            train_end = max(0, test_start - self.embargo)
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


@dataclass
class DailyRollingWalkForward:
    """
    매 봉마다 학습·예측을 전진시키는 검증기.

    각 fold:
        학습 = [train_start, t - embargo)
        검증 = [t, t + test_window)

    max_train_size 가 None 이면 누적식, 정수면 고정식 sliding.
    """
    min_train: int = 252
    embargo: int = 20
    step: int = 1
    test_window: int = 1
    max_train_size: int | None = None

    def split(self, n_rows: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        t = self.min_train
        while t + self.test_window <= n_rows:
            train_end = max(0, t - self.embargo)
            if train_end <= 0:
                t += self.step
                continue
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            yield np.arange(train_start, train_end), np.arange(t, t + self.test_window)
            t += self.step

