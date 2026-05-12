import numpy as np

from fibtrader import BotConfig, add_features, attach_recent_swing_and_fib, zigzag_confirmed
from fibtrader.ml import (
    DailyRollingWalkForward,
    build_dataset,
    build_failure_notebook,
    cluster_failures,
    train_xgb_rolling,
    triple_barrier_labels,
)


def _pipeline(df, cfg):
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    feat = add_features(sw, cfg)
    return triple_barrier_labels(feat, cfg)


def test_daily_rolling_split_basic():
    splitter = DailyRollingWalkForward(min_train=100, embargo=20, step=1, test_window=1)
    n = 200
    seen_tests = []
    for tr, te in splitter.split(n):
        assert len(te) == 1
        assert tr.max() < te.min() - 19
        seen_tests.append(int(te[0]))
    # step=1 -> contiguous
    assert seen_tests == list(range(100, 200))


def test_daily_rolling_respects_step():
    splitter = DailyRollingWalkForward(min_train=100, embargo=10, step=5, test_window=1)
    seen = [int(te[0]) for _, te in splitter.split(200)]
    assert seen[0] == 100
    diffs = np.diff(seen)
    assert (diffs == 5).all()


def test_train_xgb_rolling_runs(synth_df):
    cfg = BotConfig(zigzag_pct=0.05)
    feat = _pipeline(synth_df, cfg)
    X, y, idx = build_dataset(feat)
    if len(X) < 300:
        return
    result = train_xgb_rolling(
        X, y,
        min_train=200, embargo=cfg.hold_bars, step=5, refit_every=20,
        xgb_params={"n_estimators": 30, "max_depth": 3},
    )
    assert result.final_model is not None
    assert result.aggregate_metrics.get("n_oof", 0) > 0


def test_failure_notebook_columns(synth_df):
    cfg = BotConfig(zigzag_pct=0.05)
    feat = _pipeline(synth_df, cfg)
    X, y, idx = build_dataset(feat)
    if len(X) < 300:
        return
    result = train_xgb_rolling(
        X, y,
        min_train=200, embargo=cfg.hold_bars, step=10, refit_every=20,
        xgb_params={"n_estimators": 30, "max_depth": 3},
    )
    nb = build_failure_notebook(
        feat, result.oof_proba, result.final_model, X,
        threshold=0.5, top_k_shap=3,
    )
    assert {"mistake_type", "oof_proba", "label", "top_pos_features", "top_neg_features"}.issubset(nb.columns)
    assert set(nb["mistake_type"].unique()).issubset({"TP", "TN", "FP", "FN"})


def test_cluster_failures_returns_cluster_ids(synth_df):
    cfg = BotConfig(zigzag_pct=0.05)
    feat = _pipeline(synth_df, cfg)
    X, y, idx = build_dataset(feat)
    if len(X) < 400:
        return
    result = train_xgb_rolling(
        X, y, min_train=200, embargo=cfg.hold_bars, step=5, refit_every=20,
        xgb_params={"n_estimators": 30, "max_depth": 3},
    )
    nb = build_failure_notebook(feat, result.oof_proba, result.final_model, X)
    nb = cluster_failures(nb, n_clusters=3, mistake_type="FP")
    fp = nb[nb["mistake_type"] == "FP"]
    if len(fp) >= 3:
        assert fp["cluster_id"].notna().all()
        assert set(fp["cluster_id"].unique()).issubset({0, 1, 2})
