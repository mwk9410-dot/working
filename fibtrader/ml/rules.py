"""
Rule proposer (human-in-the-loop).

This module is intentionally *not* a closed-loop auto-applier. It produces
candidate predicates that would have blocked a meaningful share of FPs at
acceptable cost to TPs, scored on the SAME notebook they were derived from.
A user must explicitly accept rules; `apply_rules` then enforces them on
future feature frames.

Method: fit a shallow decision tree on (FP=1 vs TP=0) using a numeric
feature subset. Walk each leaf, keep those with FP-purity above a threshold,
serialize the path into a pandas-eval string.

Key invariants:
  - Tree depth capped (default 2) — deeper rules almost always overfit.
  - Each candidate carries blocked_fp, blocked_tp, fp_tp_ratio and
    blocked_avg_return so the user can decide.
  - No rule is auto-activated. `apply_rules` requires explicit `enabled=True`.

Rule removal: the registry stores per-rule cumulative stats; call
`recommend_removals` to flag rules that no longer pay their cost.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NUMERIC_RULE_FEATURES = [
    "rsi", "atr_pct", "macd_hist", "vol_z",
    "confluence_count", "retracement_depth", "nearest_fib_dist",
]
BINARY_RULE_FEATURES = ["trend_up", "regime_ok", "fib_near"]


@dataclass
class CandidateRule:
    name: str
    predicate: str             # human-readable, eg "rsi > 65 AND trend_up == 0"
    py_expr: str               # pandas .eval-compatible boolean expression
    n_blocked_fp: int
    n_blocked_tp: int
    fp_tp_ratio: float         # blocked_fp / max(1, blocked_tp)
    blocked_avg_return: float  # mean forward_return of rows the rule blocks
    enabled: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _format_threshold(feature: str, threshold: float, op: str) -> Tuple[str, str]:
    if feature in BINARY_RULE_FEATURES:
        # sklearn uses <= 0.5 to split binaries
        cmp_val = 0 if threshold <= 0.5 else 1
        if op == "<=":
            return f"{feature} == {cmp_val}", f"({feature} == {cmp_val})"
        return f"{feature} == {1 - cmp_val}", f"({feature} == {1 - cmp_val})"
    rounded = round(threshold, 4)
    if op == "<=":
        return f"{feature} <= {rounded}", f"({feature} <= {rounded})"
    return f"{feature} > {rounded}", f"({feature} > {rounded})"


def _walk_tree(tree, feature_names: List[str]) -> List[List[Tuple[str, float, str]]]:
    """Return one path per leaf, each path = list of (feature, threshold, op)."""
    paths = []

    def recurse(node: int, path: List[Tuple[str, float, str]]):
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:  # leaf
            paths.append(path)
            return
        feature = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        recurse(left, path + [(feature, threshold, "<=")])
        recurse(right, path + [(feature, threshold, ">")])

    recurse(0, [])
    return paths


def propose_rules(
    notebook: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    max_depth: int = 2,
    min_fp_blocked: int = 5,
    min_fp_tp_ratio: float = 2.0,
    target_fp_purity: float = 0.7,
) -> List[CandidateRule]:
    """
    Fit a shallow tree on FP (1) vs TP (0) and extract leaves with high FP
    purity as candidate rules.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as e:
        raise ImportError("scikit-learn is required") from e

    if "mistake_type" not in notebook.columns:
        raise ValueError("notebook must include 'mistake_type'")

    fp = notebook[notebook["mistake_type"] == "FP"]
    tp = notebook[notebook["mistake_type"] == "TP"]
    if len(fp) < min_fp_blocked or len(tp) < 5:
        return []

    if feature_cols is None:
        feature_cols = [c for c in NUMERIC_RULE_FEATURES + BINARY_RULE_FEATURES
                        if c in notebook.columns]
    if not feature_cols:
        return []

    labels = np.concatenate([np.ones(len(fp)), np.zeros(len(tp))])
    X = pd.concat([fp[feature_cols], tp[feature_cols]], axis=0).fillna(0.0)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(5, min_fp_blocked),
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, labels)

    paths = _walk_tree(clf.tree_, feature_cols)
    candidates: List[CandidateRule] = []

    for i, path in enumerate(paths):
        if not path:
            continue
        # Build mask on the *full* notebook to evaluate against all classes
        masks = []
        readable_parts = []
        expr_parts = []
        for feature, thr, op in path:
            human, expr = _format_threshold(feature, thr, op)
            readable_parts.append(human)
            expr_parts.append(expr)
            if feature in BINARY_RULE_FEATURES:
                cmp_val = 0 if thr <= 0.5 else 1
                if op == "<=":
                    masks.append(notebook[feature] == cmp_val)
                else:
                    masks.append(notebook[feature] == (1 - cmp_val))
            else:
                if op == "<=":
                    masks.append(notebook[feature] <= round(thr, 4))
                else:
                    masks.append(notebook[feature] > round(thr, 4))
        m = masks[0]
        for extra in masks[1:]:
            m = m & extra

        blocked = notebook[m]
        blocked_fp = int((blocked["mistake_type"] == "FP").sum())
        blocked_tp = int((blocked["mistake_type"] == "TP").sum())
        if blocked_fp < min_fp_blocked:
            continue
        purity = blocked_fp / max(1, blocked_fp + blocked_tp)
        if purity < target_fp_purity:
            continue
        ratio = blocked_fp / max(1, blocked_tp)
        if ratio < min_fp_tp_ratio:
            continue
        avg_ret = float(blocked["forward_return"].mean()) if "forward_return" in blocked.columns else float("nan")

        predicate = " AND ".join(readable_parts)
        py_expr = " and ".join(expr_parts)
        candidates.append(CandidateRule(
            name=f"rule_{i:02d}",
            predicate=predicate,
            py_expr=py_expr,
            n_blocked_fp=blocked_fp,
            n_blocked_tp=blocked_tp,
            fp_tp_ratio=round(ratio, 3),
            blocked_avg_return=round(avg_ret, 5) if np.isfinite(avg_ret) else float("nan"),
            enabled=False,
        ))

    candidates.sort(key=lambda r: (r.fp_tp_ratio, r.n_blocked_fp), reverse=True)
    return candidates


def save_rules(rules: List[CandidateRule], path: str | Path) -> None:
    Path(path).write_text(json.dumps([r.to_dict() for r in rules], indent=2))


def load_rules(path: str | Path) -> List[CandidateRule]:
    data = json.loads(Path(path).read_text())
    return [CandidateRule(**r) for r in data]


def apply_rules(
    df: pd.DataFrame,
    rules: List[CandidateRule],
    only_enabled: bool = True,
) -> pd.Series:
    """
    Return a boolean Series — True means "blocked by at least one rule".
    Disabled rules are ignored when only_enabled=True (default).
    Use ~blocked as the entry mask to forbid those rows.
    """
    blocked = pd.Series(False, index=df.index)
    for r in rules:
        if only_enabled and not r.enabled:
            continue
        try:
            mask = df.eval(r.py_expr)
        except Exception:
            continue
        blocked = blocked | mask.fillna(False).astype(bool)
    return blocked


# -- Rule registry / removal recommendation -----------------------------

@dataclass
class RuleStats:
    name: str
    cumulative_blocked_fp: int = 0
    cumulative_blocked_tp: int = 0
    last_seen_ratio: float = float("nan")
    epochs_active: int = 0
    epochs_silent: int = 0      # rule blocked nothing in this epoch


def update_registry(
    registry: Dict[str, RuleStats],
    rules: List[CandidateRule],
    notebook: pd.DataFrame,
) -> Dict[str, RuleStats]:
    """Update cumulative stats for each enabled rule on the latest notebook."""
    for r in rules:
        if not r.enabled:
            continue
        stats = registry.setdefault(r.name, RuleStats(name=r.name))
        stats.epochs_active += 1
        try:
            mask = notebook.eval(r.py_expr).fillna(False).astype(bool)
        except Exception:
            stats.epochs_silent += 1
            continue
        blocked = notebook[mask]
        bfp = int((blocked["mistake_type"] == "FP").sum())
        btp = int((blocked["mistake_type"] == "TP").sum())
        stats.cumulative_blocked_fp += bfp
        stats.cumulative_blocked_tp += btp
        if bfp + btp == 0:
            stats.epochs_silent += 1
        stats.last_seen_ratio = bfp / max(1, btp)
    return registry


def recommend_removals(
    registry: Dict[str, RuleStats],
    min_ratio: float = 1.5,
    max_silent_epochs: int = 3,
) -> List[str]:
    """
    Suggest rules to retire:
      - cumulative FP/TP ratio < min_ratio (rule hurts more than it helps), or
      - silent for too many epochs (no longer triggers).
    """
    out = []
    for name, s in registry.items():
        if s.cumulative_blocked_tp == 0 and s.cumulative_blocked_fp == 0:
            if s.epochs_silent >= max_silent_epochs:
                out.append(name)
            continue
        ratio = s.cumulative_blocked_fp / max(1, s.cumulative_blocked_tp)
        if ratio < min_ratio:
            out.append(name)
        elif s.epochs_silent >= max_silent_epochs:
            out.append(name)
    return out
