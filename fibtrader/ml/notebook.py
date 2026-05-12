"""
Failure-case notebook ("오답노트").

Joins out-of-sample predictions with raw features, triple-barrier outcomes, and
per-row SHAP attributions so each misclassified bar carries enough context to
diagnose WHY the model was wrong.

For each row we record:
    mistake_type        TP / FP / FN / TN
    proba               P(win) from OOS
    label               actual triple-barrier outcome (0/1)
    forward_return      realized return at exit
    barrier_touched     "stop" | "take" | "time"
    top_pos_features    top-K features pushing proba UP at this row (SHAP)
    top_neg_features    top-K features pushing proba DOWN at this row (SHAP)
    context columns     a curated subset of indicators for human review

FP rows (model said "win" but lost real money) are the highest-priority for
post-mortem, so a convenience filter is provided.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = [
    "Date",
    "Close",
    "atr_pct",
    "rsi",
    "macd_hist",
    "trend_up",
    "regime_ok",
    "confluence_count",
    "fib_near",
    "nearest_fib_level",
    "nearest_fib_dist",
    "retracement_depth",
    "vol_z",
]


def _classify(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 1 and pred == 0:
        return "FN"
    if label == 0 and pred == 1:
        return "FP"
    return "TN"


def build_failure_notebook(
    feat_df: pd.DataFrame,
    oof_proba: pd.Series,
    model,
    X: pd.DataFrame,
    threshold: float = 0.5,
    top_k_shap: int = 5,
    only_failures: bool = False,
    max_shap_rows: int = 10000,
) -> pd.DataFrame:
    """
    Build a per-row notebook of OOS decisions and their outcomes.

    Parameters
    ----------
    feat_df : full features dataframe (must include `label`, `forward_return`,
              `barrier_touched`, and the columns in CONTEXT_COLUMNS that exist).
    oof_proba : OOS probabilities (NaN for unscored bars).
    model : trained xgboost classifier (used for per-row SHAP).
    X : feature matrix that was fed to the model (same columns).
    threshold : decision threshold for proba -> pred.
    top_k_shap : how many features to record per direction.
    only_failures : if True, return only FP and FN rows.
    max_shap_rows : SHAP cost cap; if scored rows exceed this we still cover all
                    failure rows plus a random sample of correct rows.
    """
    df = feat_df.copy()
    df["oof_proba"] = oof_proba
    scored = df["oof_proba"].notna() & df["label"].notna()
    nb = df.loc[scored].copy()

    nb["oof_pred"] = (nb["oof_proba"] >= threshold).astype(int)
    nb["mistake_type"] = [
        _classify(int(l), int(p)) for l, p in zip(nb["label"], nb["oof_pred"])
    ]

    keep_ctx = [c for c in CONTEXT_COLUMNS if c in nb.columns]
    base_cols = ["oof_proba", "oof_pred", "label", "mistake_type",
                 "forward_return", "barrier_touched"] + keep_ctx
    nb = nb[base_cols]

    # SHAP per row, scoped to whatever we actually need
    try:
        import shap
    except ImportError:
        nb["top_pos_features"] = ""
        nb["top_neg_features"] = ""
        return nb.loc[nb["mistake_type"].isin({"FP", "FN"})] if only_failures else nb

    X_scored = X.loc[nb.index]
    n_scored = len(X_scored)

    if n_scored > max_shap_rows:
        # Always include all failures; subsample TPs/TNs to fit the budget.
        fail_mask = nb["mistake_type"].isin({"FP", "FN"})
        fail_idx = nb.index[fail_mask]
        correct_idx = nb.index[~fail_mask]
        budget = max_shap_rows - len(fail_idx)
        if budget > 0 and len(correct_idx) > budget:
            sampled_correct = pd.Index(np.random.default_rng(0).choice(correct_idx, size=budget, replace=False))
        else:
            sampled_correct = correct_idx
        target_idx = fail_idx.union(sampled_correct)
        X_shap = X.loc[target_idx]
    else:
        target_idx = X_scored.index
        X_shap = X_scored

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_df = pd.DataFrame(shap_vals, index=X_shap.index, columns=X_shap.columns)

    def _topk(row: pd.Series, k: int, positive: bool) -> str:
        sorted_idx = row.sort_values(ascending=not positive).index
        picks = []
        for col in sorted_idx[:k]:
            v = row[col]
            if (positive and v <= 0) or (not positive and v >= 0):
                break
            picks.append(f"{col}={v:+.3f}")
        return ";".join(picks)

    pos_series = shap_df.apply(lambda r: _topk(r, top_k_shap, positive=True), axis=1)
    neg_series = shap_df.apply(lambda r: _topk(r, top_k_shap, positive=False), axis=1)
    nb["top_pos_features"] = pos_series.reindex(nb.index).fillna("")
    nb["top_neg_features"] = neg_series.reindex(nb.index).fillna("")

    if only_failures:
        nb = nb.loc[nb["mistake_type"].isin({"FP", "FN"})]
    return nb


def cluster_failures(
    notebook: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 5,
    mistake_type: str = "FP",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    KMeans-cluster failure rows on a numeric feature subset and return the
    notebook with a `cluster_id` column added (NaN for non-failure rows).

    Defaults: cluster FP rows on the curated context indicators.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from e

    nb = notebook.copy()
    nb["cluster_id"] = np.nan

    target = nb[nb["mistake_type"] == mistake_type]
    if len(target) < n_clusters:
        return nb

    if feature_cols is None:
        feature_cols = [c for c in CONTEXT_COLUMNS
                        if c in target.columns
                        and pd.api.types.is_numeric_dtype(target[c])]
    if not feature_cols:
        return nb

    Xf = target[feature_cols].fillna(target[feature_cols].median())
    if Xf.empty or len(Xf.columns) == 0:
        return nb
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xf)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)
    nb.loc[target.index, "cluster_id"] = labels
    return nb
