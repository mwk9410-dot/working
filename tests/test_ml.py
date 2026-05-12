import numpy as np
import pandas as pd

from fibtrader import BotConfig, add_features, attach_recent_swing_and_fib, zigzag_confirmed
from fibtrader.ml import (
    EmbargoedWalkForward,
    build_dataset,
    train_xgb_walkforward,
    triple_barrier_labels,
)


def _pipeline(df, cfg):
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    feat = add_features(sw, cfg)
    return triple_barrier_labels(feat, cfg)


def test_triple_barrier_label_consistency(synth_df):
    cfg = BotConfig(zigzag_pct=0.05)
    feat = _pipeline(synth_df, cfg)
    lbl = feat["label"].dropna()
    assert set(lbl.unique()).issubset({0, 1})
    # On a random walk, most labels should be 0 with positive-drift bias possibly raising
    # the win rate above ~10%. Sanity: at least some 0s exist.
    assert (lbl == 0).any()


def test_dataset_excludes_leakage_columns(synth_df):
    cfg = BotConfig()
    feat = _pipeline(synth_df, cfg)
    X, y, idx = build_dataset(feat)
    blocked = {"label", "barrier_touched", "forward_return", "forward_bars",
               "Open", "High", "Low", "Close", "Volume", "Date", "pivot"}
    assert not (set(X.columns) & blocked)
    assert len(X) == len(y) == len(idx)


def test_walkforward_split_no_overlap():
    splitter = EmbargoedWalkForward(n_splits=4, embargo=20, min_train=200)
    n = 1000
    seen_test = set()
    for tr, te in splitter.split(n):
        # No overlap between train and test
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
        # Embargo gap respected
        if len(tr) > 0:
            assert tr.max() < te.min() - 19  # at least embargo-1 gap
        # Test partitions disjoint
        for x in te.tolist():
            assert x not in seen_test
            seen_test.add(x)


def test_xgb_walkforward_runs(synth_df):
    cfg = BotConfig(zigzag_pct=0.05)
    feat = _pipeline(synth_df, cfg)
    X, y, idx = build_dataset(feat)
    if len(X) < 250:
        return
    result = train_xgb_walkforward(
        X, y, n_splits=3, embargo=cfg.hold_bars, min_train=150,
        xgb_params={"n_estimators": 50, "max_depth": 3},
    )
    assert result.final_model is not None
    assert len(result.feature_importance) == X.shape[1]
    assert len(result.fold_metrics) >= 1
