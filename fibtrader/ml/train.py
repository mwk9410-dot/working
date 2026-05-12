"""
XGBoost trainer with embargoed walk-forward.

We do NOT tune hyperparameters here; tuning belongs inside an inner CV. This
module just gives an honest first-pass OOS estimate, OOS probabilities (for
threshold sweeping), and the trained final model fit on all data for later
inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .splits import EmbargoedWalkForward


@dataclass
class TrainResult:
    oof_proba: pd.Series                          # P(label=1), aligned to X index
    fold_metrics: List[Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    feature_importance: pd.Series                 # gain-based from final model
    final_model: object = None                    # xgb.XGBClassifier
    feature_names: List[str] = field(default_factory=list)


def _binary_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    # ROC AUC undefined for single-class folds; guard.
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = float(average_precision_score(y_true, p))
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["pos_rate"] = float(y_true.mean())
    return out


def train_xgb_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 20,
    min_train: int = 200,
    xgb_params: Optional[Dict] = None,
    threshold: float = 0.5,
) -> TrainResult:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    default_params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    params = {**default_params, **(xgb_params or {})}

    splitter = EmbargoedWalkForward(n_splits=n_splits, embargo=embargo, min_train=min_train)
    n_rows = len(X)
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    oof = np.full(n_rows, np.nan)
    fold_metrics = []

    for train_idx, test_idx in splitter.split(n_rows):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_arr[train_idx], y_arr[train_idx])
        p = clf.predict_proba(X_arr[test_idx])[:, 1]
        oof[test_idx] = p
        fold_metrics.append(_binary_metrics(y_arr[test_idx], p, threshold=threshold))

    valid = ~np.isnan(oof)
    if valid.any():
        agg = _binary_metrics(y_arr[valid], oof[valid], threshold=threshold)
        agg["folds"] = float(len(fold_metrics))
        agg["n_oof"] = float(int(valid.sum()))
    else:
        agg = {"folds": 0.0, "n_oof": 0.0}

    final = xgb.XGBClassifier(**params)
    final.fit(X_arr, y_arr)
    importance = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)

    return TrainResult(
        oof_proba=pd.Series(oof, index=X.index, name="oof_proba"),
        fold_metrics=fold_metrics,
        aggregate_metrics=agg,
        feature_importance=importance,
        final_model=final,
        feature_names=list(X.columns),
    )
