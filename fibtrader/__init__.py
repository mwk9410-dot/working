from .config import BotConfig
from .swing import zigzag_confirmed
from .fib import fib_levels_from_leg, attach_recent_swing_and_fib
from .features import add_features
from .backtest import backtest_fib_confluence
from .stats import summarize_trades, permutation_test_random_ratios
from .signal import LiveSignalGenerator
from .pipeline import prepare_features

__all__ = [
    "BotConfig",
    "zigzag_confirmed",
    "fib_levels_from_leg",
    "attach_recent_swing_and_fib",
    "add_features",
    "backtest_fib_confluence",
    "summarize_trades",
    "permutation_test_random_ratios",
    "LiveSignalGenerator",
    "prepare_features",
]

# Subpackages are imported lazily — they pull heavy deps (xgboost, shap).

