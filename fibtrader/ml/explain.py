"""SHAP-based feature importance for the trained XGBoost model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def shap_importance(
    model,
    X: pd.DataFrame,
    sample: Optional[int] = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by feature name with columns:
        mean_abs_shap     : average |SHAP|, the canonical importance summary
        mean_shap         : signed average SHAP (direction of contribution)
    Sampling is used for large X because TreeExplainer over many rows is slow
    and contributes diminishing accuracy beyond a few thousand rows.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is required: pip install shap") from e

    if sample is not None and len(X) > sample:
        X_use = X.sample(n=sample, random_state=random_state)
    else:
        X_use = X

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_use)
    # xgboost binary classifier returns 2D array (n_samples, n_features)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    abs_mean = np.abs(shap_vals).mean(axis=0)
    signed_mean = shap_vals.mean(axis=0)
    out = pd.DataFrame(
        {"mean_abs_shap": abs_mean, "mean_shap": signed_mean},
        index=X_use.columns,
    ).sort_values("mean_abs_shap", ascending=False)
    return out
