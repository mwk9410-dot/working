import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


def _synthetic_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Random walk with a slow upward drift and occasional sharp swings
    rets = rng.normal(0.0008, 0.012, size=n)
    rets[100:110] += 0.02
    rets[250:260] -= 0.025
    rets[400:410] += 0.015
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.001, 0.015, size=n))
    low = close * (1 - rng.uniform(0.001, 0.015, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000, 10_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "Open": open_,
            "High": np.maximum.reduce([high, open_, close]),
            "Low": np.minimum.reduce([low, open_, close]),
            "Close": close,
            "Volume": volume,
        }
    )


@pytest.fixture
def synth_df():
    return _synthetic_ohlcv()
