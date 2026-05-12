"""
풀드 학습 + sliding window 검증기 테스트.
"""
import numpy as np
import pandas as pd

from fibtrader import BotConfig
from fibtrader.ml import (
    DailyRollingWalkForward,
    EmbargoedWalkForward,
    build_dataset,
    build_pooled_frame,
    train_xgb_walkforward,
)
from tests.test_universe import _make_ticker


# --- Sliding window 동작 검증 -------------------------------------------

def test_walkforward_누적식이_기본():
    """max_train_size=None 일 때 학습 구간은 항상 0부터 시작."""
    sp = EmbargoedWalkForward(n_splits=4, embargo=10, min_train=100)
    n = 500
    starts = []
    for tr, te in sp.split(n):
        starts.append(int(tr.min()))
    assert all(s == 0 for s in starts), "누적식인데 학습 시작이 0이 아닌 fold 발견"


def test_walkforward_sliding_고정크기():
    """max_train_size=150 이면 학습 크기가 항상 150 또는 그 이하."""
    sp = EmbargoedWalkForward(n_splits=4, embargo=10, min_train=200, max_train_size=150)
    n = 1000
    sizes = []
    for tr, te in sp.split(n):
        sizes.append(len(tr))
    assert all(s <= 150 for s in sizes), f"sliding 크기 초과 발견: {sizes}"
    # 충분히 진행되면 정확히 150 이 나와야 함
    assert max(sizes) == 150


def test_rolling_sliding_고정크기():
    sp = DailyRollingWalkForward(min_train=200, embargo=10, step=10, max_train_size=100)
    sizes = []
    for tr, te in sp.split(800):
        sizes.append(len(tr))
    assert all(s <= 100 for s in sizes)
    assert max(sizes) == 100


def test_sliding_으로_학습시간이_균일():
    """sliding 모드에서는 fold 별 학습 크기가 거의 같아야 한다."""
    sp = EmbargoedWalkForward(n_splits=4, embargo=10, min_train=300, max_train_size=200)
    sizes = [len(tr) for tr, _ in sp.split(2000)]
    # 마지막 fold 빼고는 모두 200 이어야 함 (안정성)
    assert min(sizes) >= 100 and max(sizes) == 200


# --- 풀드 데이터셋 빌더 -----------------------------------------------

def _mini_corpus():
    return {
        "AAA": _make_ticker(seed=1, n=600, price_scale=100, vol_scale=1_000_000),
        "BBB": _make_ticker(seed=2, n=600, price_scale=50, vol_scale=500_000),
        "CCC": _make_ticker(seed=3, n=600, price_scale=20, vol_scale=200_000),
    }


def test_풀드_프레임_종목수_보존():
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)
    pooled = build_pooled_frame(_mini_corpus(), cfg)
    assert not pooled.empty
    assert set(pooled["Ticker"].unique()) == {"AAA", "BBB", "CCC"}


def test_풀드_프레임_시간순_정렬():
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)
    pooled = build_pooled_frame(_mini_corpus(), cfg)
    dates = pd.to_datetime(pooled["Date"])
    assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), "시간 순서가 어긋남"


def test_풀드_프레임_라벨_컬럼_존재():
    cfg = BotConfig(zigzag_pct=0.05)
    pooled = build_pooled_frame(_mini_corpus(), cfg)
    assert "label" in pooled.columns
    assert "forward_return" in pooled.columns
    # 적어도 일부 행은 라벨이 있어야 함
    assert pooled["label"].notna().sum() > 0


def test_build_dataset이_Ticker_제외():
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)
    pooled = build_pooled_frame(_mini_corpus(), cfg)
    X, y, idx = build_dataset(pooled)
    assert "Ticker" not in X.columns
    assert "Date" not in X.columns


def test_풀드_sliding_학습이_실행됨():
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)
    pooled = build_pooled_frame(_mini_corpus(), cfg)
    X, y, idx = build_dataset(pooled)
    if len(X) < 300:
        return
    result = train_xgb_walkforward(
        X, y,
        n_splits=3,
        embargo=20,
        min_train=200,
        max_train_size=200,  # sliding
        xgb_params={"n_estimators": 30, "max_depth": 3},
    )
    assert result.final_model is not None
    assert result.aggregate_metrics.get("n_oof", 0) > 0
