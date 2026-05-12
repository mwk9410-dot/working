from .labels import triple_barrier_labels
from .dataset import build_dataset, FEATURE_BLOCKLIST
from .splits import EmbargoedWalkForward, DailyRollingWalkForward
from .train import train_xgb_walkforward, train_xgb_rolling, TrainResult
from .explain import shap_importance
from .notebook import build_failure_notebook, cluster_failures
from .insight import generate_failure_insights
from .rules import (
    CandidateRule,
    RuleStats,
    apply_rules,
    load_rules,
    propose_rules,
    recommend_removals,
    save_rules,
    update_registry,
)

__all__ = [
    "triple_barrier_labels",
    "build_dataset",
    "FEATURE_BLOCKLIST",
    "EmbargoedWalkForward",
    "DailyRollingWalkForward",
    "train_xgb_walkforward",
    "train_xgb_rolling",
    "TrainResult",
    "shap_importance",
    "build_failure_notebook",
    "cluster_failures",
    "generate_failure_insights",
    "CandidateRule",
    "RuleStats",
    "propose_rules",
    "apply_rules",
    "save_rules",
    "load_rules",
    "update_registry",
    "recommend_removals",
]
