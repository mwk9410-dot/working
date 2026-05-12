"""
Automated insight extraction from a failure notebook.

Compares one mistake class (default: FP) against a reference class
(default: TP + TN combined) feature-by-feature, ranks discriminating
features by effect size, and emits a human-readable text report.

Statistical tests:
  - Numeric features  : Welch's t-test  + Cohen's d
  - Binary  features  : two-proportion z-test  + odds-ratio difference
  - Categorical (str) : chi-squared        + Cramer's V

Multiple-comparison correction: Holm-Bonferroni (default on).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..stats import _cohens_d

DEFAULT_FEATURES = [
    "rsi", "atr_pct", "macd_hist", "vol_z",
    "confluence_count", "retracement_depth", "nearest_fib_dist",
    "trend_up", "regime_ok", "fib_near", "rsi_reversal",
    "macd_confirm", "bb_confirm", "vol_confirm",
    "nearest_fib_level",
]


@dataclass
class FeatureInsight:
    feature: str
    kind: str          # "numeric" | "binary" | "categorical"
    failure_summary: str
    reference_summary: str
    effect_size: float
    effect_label: str  # "Cohen's d" | "Δ proportion" | "Cramer's V"
    p_value: float
    p_value_corrected: float
    direction: str     # "↑ in FP" | "↓ in FP" | "shift"


def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return float("nan"), 1.0
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def _two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    from scipy import stats
    if n1 == 0 or n2 == 0:
        return float("nan"), 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _cramers_v(table: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    if table.size == 0 or table.sum() == 0:
        return float("nan"), 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    v = float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")
    return v, float(p)


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    corrected = [float("nan")] * n
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n - rank) * p_values[idx])
        adj = max(adj, prev)
        corrected[idx] = adj
        prev = adj
    return corrected


def _detect_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        unique = series.dropna().unique()
        if len(unique) <= 2 and set(unique).issubset({0, 1, 0.0, 1.0}):
            return "binary"
        return "numeric"
    return "categorical"


def generate_failure_insights(
    notebook: pd.DataFrame,
    failure_class: str = "FP",
    reference_classes: Optional[Iterable[str]] = None,
    feature_cols: Optional[List[str]] = None,
    top_k: int = 10,
    holm_bonferroni: bool = True,
    alpha: float = 0.05,
) -> dict:
    if "mistake_type" not in notebook.columns:
        raise ValueError("notebook must include 'mistake_type'")

    if reference_classes is None:
        reference_classes = [c for c in ("TP", "TN") if c in notebook["mistake_type"].unique()]
    reference_classes = list(reference_classes)

    fail_df = notebook[notebook["mistake_type"] == failure_class]
    ref_df = notebook[notebook["mistake_type"].isin(reference_classes)]
    if fail_df.empty or ref_df.empty:
        return {
            "summary": pd.DataFrame(),
            "report": f"insufficient data: |{failure_class}|={len(fail_df)}, |ref|={len(ref_df)}",
            "meta": {"n_failure": len(fail_df), "n_reference": len(ref_df)},
        }

    cols = feature_cols if feature_cols is not None else [
        c for c in DEFAULT_FEATURES if c in notebook.columns
    ]

    raw: List[FeatureInsight] = []
    p_values: List[float] = []

    for col in cols:
        kind = _detect_kind(notebook[col])
        if kind == "numeric":
            a = fail_df[col].dropna().to_numpy()
            b = ref_df[col].dropna().to_numpy()
            d = _cohens_d(a, b)
            _, p = _welch_t(a, b)
            if not np.isfinite(d):
                continue
            direction = "↑ in FP" if d > 0 else ("↓ in FP" if d < 0 else "no shift")
            fs = f"{a.mean():.4g} ± {a.std(ddof=1) if len(a) > 1 else 0:.4g} (n={len(a)})"
            rs = f"{b.mean():.4g} ± {b.std(ddof=1) if len(b) > 1 else 0:.4g} (n={len(b)})"
            raw.append(FeatureInsight(col, "numeric", fs, rs, d, "Cohen's d", p, np.nan, direction))
            p_values.append(p)

        elif kind == "binary":
            a = fail_df[col].dropna().astype(int)
            b = ref_df[col].dropna().astype(int)
            k1, n1 = int(a.sum()), len(a)
            k2, n2 = int(b.sum()), len(b)
            if n1 == 0 or n2 == 0:
                continue
            p1, p2 = k1 / n1, k2 / n2
            _, p = _two_prop_z(k1, n1, k2, n2)
            delta = p1 - p2
            direction = "↑ in FP" if delta > 0 else ("↓ in FP" if delta < 0 else "no shift")
            fs = f"{p1:.1%} (k={k1}/{n1})"
            rs = f"{p2:.1%} (k={k2}/{n2})"
            raw.append(FeatureInsight(col, "binary", fs, rs, delta, "Δ proportion", p, np.nan, direction))
            p_values.append(p)

        else:  # categorical
            a_counts = fail_df[col].value_counts()
            b_counts = ref_df[col].value_counts()
            categories = sorted(set(a_counts.index) | set(b_counts.index))
            if not categories:
                continue
            table = np.array([
                [int(a_counts.get(c, 0)) for c in categories],
                [int(b_counts.get(c, 0)) for c in categories],
            ])
            v, p = _cramers_v(table)
            if not np.isfinite(v):
                continue
            top_a = a_counts.idxmax() if not a_counts.empty else "?"
            top_b = b_counts.idxmax() if not b_counts.empty else "?"
            fs = f"mode={top_a} ({a_counts.max()/a_counts.sum():.0%})"
            rs = f"mode={top_b} ({b_counts.max()/b_counts.sum():.0%})"
            raw.append(FeatureInsight(col, "categorical", fs, rs, v, "Cramer's V", p, np.nan, "shift"))
            p_values.append(p)

    if not raw:
        return {
            "summary": pd.DataFrame(),
            "report": "no testable features",
            "meta": {"n_failure": len(fail_df), "n_reference": len(ref_df)},
        }

    corrected = _holm_bonferroni(p_values) if holm_bonferroni else p_values
    for ins, pc in zip(raw, corrected):
        ins.p_value_corrected = pc

    summary = pd.DataFrame([{
        "feature": r.feature,
        "kind": r.kind,
        "effect_size": r.effect_size,
        "effect_label": r.effect_label,
        "direction": r.direction,
        "p_value": r.p_value,
        "p_value_corrected": r.p_value_corrected,
        "failure": r.failure_summary,
        "reference": r.reference_summary,
    } for r in raw])
    summary["abs_effect"] = summary["effect_size"].abs()
    summary = summary.sort_values("abs_effect", ascending=False).drop(columns=["abs_effect"]).reset_index(drop=True)

    sig_mask = summary["p_value_corrected"] <= alpha
    sig = summary[sig_mask].head(top_k)
    up = sig[sig["direction"] == "↑ in FP"]
    down = sig[sig["direction"] == "↓ in FP"]
    shift = sig[sig["direction"] == "shift"]

    lines = []
    lines.append(f"=== {failure_class} Insight Report "
                 f"({len(fail_df)} cases vs {len(ref_df)} reference) ===")
    lines.append(f"Tested {len(summary)} features. "
                 f"{int(sig_mask.sum())} significant at α={alpha}"
                 f"{' (Holm-Bonferroni)' if holm_bonferroni else ''}.")
    lines.append("")

    if not up.empty:
        lines.append(f"[Significantly elevated in {failure_class}]")
        for _, r in up.iterrows():
            lines.append(f"  {r['feature']:<22s} {r['failure']}  vs ref {r['reference']}  "
                         f"({r['effect_label']}={r['effect_size']:+.3f}, p={r['p_value_corrected']:.3g})")
        lines.append("")
    if not down.empty:
        lines.append(f"[Significantly depressed in {failure_class}]")
        for _, r in down.iterrows():
            lines.append(f"  {r['feature']:<22s} {r['failure']}  vs ref {r['reference']}  "
                         f"({r['effect_label']}={r['effect_size']:+.3f}, p={r['p_value_corrected']:.3g})")
        lines.append("")
    if not shift.empty:
        lines.append("[Distribution shift (categorical)]")
        for _, r in shift.iterrows():
            lines.append(f"  {r['feature']:<22s} FP {r['failure']}  vs ref {r['reference']}  "
                         f"(V={r['effect_size']:.3f}, p={r['p_value_corrected']:.3g})")
        lines.append("")

    if sig.empty:
        lines.append("(no features pass significance after correction — failures look noise-driven)")

    report = "\n".join(lines)
    return {
        "summary": summary,
        "report": report,
        "meta": {
            "n_failure": int(len(fail_df)),
            "n_reference": int(len(ref_df)),
            "n_tested": int(len(summary)),
            "n_significant": int(sig_mask.sum()),
            "alpha": float(alpha),
        },
    }
