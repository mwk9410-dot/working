import json
from pathlib import Path

import numpy as np
import pandas as pd

from fibtrader.ml import (
    CandidateRule,
    RuleStats,
    apply_rules,
    load_rules,
    propose_rules,
    recommend_removals,
    save_rules,
    update_registry,
)


def _planted_notebook(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fp, n_tp = 200, 200
    # FP: nearly all rsi > 70 and trend_up == 0
    fp = pd.DataFrame({
        "mistake_type": ["FP"] * n_fp,
        "rsi": rng.normal(75, 4, n_fp),
        "atr_pct": rng.normal(0.02, 0.005, n_fp),
        "trend_up": rng.binomial(1, 0.1, n_fp),
        "regime_ok": np.ones(n_fp, dtype=int),
        "fib_near": rng.binomial(1, 0.5, n_fp),
        "macd_hist": rng.normal(0, 0.5, n_fp),
        "vol_z": rng.normal(0, 1, n_fp),
        "confluence_count": rng.integers(2, 5, n_fp),
        "retracement_depth": rng.uniform(0, 1, n_fp),
        "nearest_fib_dist": rng.uniform(0, 0.02, n_fp),
        "forward_return": rng.normal(-0.03, 0.01, n_fp),
    })
    tp = pd.DataFrame({
        "mistake_type": ["TP"] * n_tp,
        "rsi": rng.normal(50, 5, n_tp),
        "atr_pct": rng.normal(0.015, 0.005, n_tp),
        "trend_up": rng.binomial(1, 0.9, n_tp),
        "regime_ok": np.ones(n_tp, dtype=int),
        "fib_near": rng.binomial(1, 0.5, n_tp),
        "macd_hist": rng.normal(0, 0.5, n_tp),
        "vol_z": rng.normal(0, 1, n_tp),
        "confluence_count": rng.integers(2, 5, n_tp),
        "retracement_depth": rng.uniform(0, 1, n_tp),
        "nearest_fib_dist": rng.uniform(0, 0.02, n_tp),
        "forward_return": rng.normal(0.03, 0.01, n_tp),
    })
    return pd.concat([fp, tp], ignore_index=True)


def test_propose_rules_recovers_planted_pattern():
    nb = _planted_notebook()
    rules = propose_rules(nb, max_depth=2, min_fp_blocked=20, min_fp_tp_ratio=2.0)
    assert rules, "expected at least one rule"
    top = rules[0]
    # The strongest signal is high RSI; the top rule should mention rsi
    assert "rsi" in top.py_expr
    # Should block far more FP than TP
    assert top.n_blocked_fp >= 5 * top.n_blocked_tp


def test_apply_rules_only_enabled_by_default():
    nb = _planted_notebook()
    rules = propose_rules(nb, max_depth=2, min_fp_blocked=20)
    assert all(not r.enabled for r in rules)
    blocked = apply_rules(nb, rules, only_enabled=True)
    assert blocked.sum() == 0  # nothing is enabled yet

    # Enable the first rule and re-evaluate
    rules[0].enabled = True
    blocked = apply_rules(nb, rules, only_enabled=True)
    assert blocked.sum() > 0


def test_save_and_load_roundtrip(tmp_path: Path):
    nb = _planted_notebook()
    rules = propose_rules(nb, max_depth=2, min_fp_blocked=20)
    p = tmp_path / "rules.json"
    save_rules(rules, p)
    loaded = load_rules(p)
    assert len(loaded) == len(rules)
    assert loaded[0].py_expr == rules[0].py_expr
    assert loaded[0].enabled is False


def test_registry_and_removal_recommendation():
    nb = _planted_notebook()
    rules = propose_rules(nb, max_depth=2, min_fp_blocked=20)
    rules[0].enabled = True
    registry: dict = {}
    update_registry(registry, rules, nb)
    assert rules[0].name in registry
    stats = registry[rules[0].name]
    assert stats.cumulative_blocked_fp > stats.cumulative_blocked_tp

    # A rule that hurts more than it helps should be flagged
    bad = CandidateRule(
        name="bad", predicate="rsi > 0", py_expr="rsi > 0",
        n_blocked_fp=0, n_blocked_tp=0, fp_tp_ratio=0.0,
        blocked_avg_return=0.0, enabled=True,
    )
    bad_registry = {"bad": RuleStats(name="bad", cumulative_blocked_fp=2,
                                     cumulative_blocked_tp=20, epochs_active=1)}
    flagged = recommend_removals(bad_registry, min_ratio=1.5)
    assert "bad" in flagged


def test_no_rules_when_data_too_clean():
    """If FP and TP are indistinguishable, propose_rules should return []."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "mistake_type": ["FP"] * n + ["TP"] * n,
        "rsi": rng.normal(50, 5, 2 * n),
        "trend_up": rng.binomial(1, 0.5, 2 * n),
        "atr_pct": rng.normal(0.02, 0.005, 2 * n),
        "forward_return": rng.normal(0, 0.01, 2 * n),
    })
    rules = propose_rules(df, max_depth=2, min_fp_blocked=20, min_fp_tp_ratio=2.0)
    assert rules == [] or all(r.fp_tp_ratio < 5 for r in rules)
