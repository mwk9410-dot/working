import numpy as np
import pandas as pd
import pytest

from fibtrader import BotConfig
from fibtrader.universe import (
    FilterSpec,
    backtest_corpus,
    compute_ticker_stats,
    sensitivity_sweep,
)


def _make_ticker(seed: int, n: int = 800, price_scale: float = 50.0, vol_scale: int = 5000):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.015, size=n)
    close = price_scale * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.001, 0.012, size=n))
    low = close * (1 - rng.uniform(0.001, 0.012, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(vol_scale // 4, vol_scale, size=n).astype(float)
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "Open": open_,
        "High": np.maximum.reduce([high, open_, close]),
        "Low": np.minimum.reduce([low, open_, close]),
        "Close": close,
        "Volume": volume,
    })


@pytest.fixture
def mini_corpus():
    return {
        "BIG":  _make_ticker(seed=1, n=800, price_scale=200, vol_scale=2_000_000),
        "MID":  _make_ticker(seed=2, n=800, price_scale=50,  vol_scale=500_000),
        "SMALL":_make_ticker(seed=3, n=800, price_scale=5,   vol_scale=50_000),
        "PENNY":_make_ticker(seed=4, n=800, price_scale=0.5, vol_scale=10_000),
        "SHORT":_make_ticker(seed=5, n=200, price_scale=100, vol_scale=1_000_000),
    }


def test_ticker_stats_basic(mini_corpus):
    stats = compute_ticker_stats(mini_corpus)
    assert set(stats.index) == set(mini_corpus.keys())
    assert (stats["n_bars"] > 0).all()
    assert stats.loc["BIG", "median_close"] > stats.loc["PENNY", "median_close"]
    assert stats.loc["BIG", "median_dollar_volume"] > stats.loc["SMALL", "median_dollar_volume"]


def test_filter_spec_drops_small_short(mini_corpus):
    stats = compute_ticker_stats(mini_corpus)
    spec = FilterSpec(min_bars=500, min_dollar_volume=1e5, min_close=1.0)
    kept = set(spec.apply(stats))
    assert "BIG" in kept
    assert "SHORT" not in kept     # too few bars
    assert "PENNY" not in kept     # price too low


def test_sensitivity_sweep_monotone_in_count(mini_corpus):
    stats = compute_ticker_stats(mini_corpus)
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)  # loose to ensure trades
    bt = backtest_corpus(mini_corpus, cfg, n_workers=1)
    sweep = sensitivity_sweep(mini_corpus, cfg, per_ticker=bt, ticker_stats=stats)
    assert not sweep.empty
    # For each dimension, raising the threshold should not increase the ticker count
    for dim in ("min_bars", "min_dollar_volume", "min_close", "min_atr_pct"):
        g = sweep[sweep["dimension"] == dim].sort_values("threshold")
        counts = g["n_tickers"].to_list()
        assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), \
            f"{dim} should be monotone non-increasing in threshold; got {counts}"


def test_backtest_corpus_returns_per_ticker(mini_corpus):
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)
    bt = backtest_corpus(mini_corpus, cfg)
    assert "ticker" in bt.columns
    assert set(bt["ticker"]).issubset(set(mini_corpus.keys()))
