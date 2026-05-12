import numpy as np
import pandas as pd

from fibtrader.ml import generate_failure_insights


def _planted_notebook(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fp, n_tp, n_tn, n_fn = 100, 100, 100, 50

    def block(label_type, rsi_mu, atr_mu, trend_up_prob, n):
        return pd.DataFrame({
            "mistake_type": [label_type] * n,
            "rsi": rng.normal(rsi_mu, 8, size=n),
            "atr_pct": rng.normal(atr_mu, 0.003, size=n),
            "trend_up": rng.binomial(1, trend_up_prob, size=n),
            "vol_z": rng.normal(0, 1, size=n),
            "nearest_fib_level": rng.choice(["dist_fib_236", "dist_fib_382", "dist_fib_500"], size=n),
        })

    # Plant: FP have RSI ~ 68, ATR% 0.025, trend_up=0 mostly.
    # TP/TN have RSI ~ 52, ATR% 0.015, trend_up=1 mostly.
    fp = block("FP", 68, 0.025, 0.25, n_fp)
    tp = block("TP", 52, 0.015, 0.80, n_tp)
    tn = block("TN", 50, 0.014, 0.78, n_tn)
    fn = block("FN", 55, 0.017, 0.70, n_fn)
    return pd.concat([fp, tp, tn, fn], ignore_index=True)


def test_insight_recovers_planted_pattern():
    nb = _planted_notebook()
    out = generate_failure_insights(nb, failure_class="FP", top_k=10)
    summary = out["summary"]

    # rsi and atr_pct must rank in the top two by effect size and be ↑ in FP
    top_two = summary.head(2)["feature"].tolist()
    assert "rsi" in top_two
    assert "atr_pct" in top_two

    rsi_row = summary[summary["feature"] == "rsi"].iloc[0]
    assert rsi_row["direction"] == "↑ in FP"
    assert rsi_row["p_value_corrected"] < 0.01

    trend_row = summary[summary["feature"] == "trend_up"].iloc[0]
    assert trend_row["direction"] == "↓ in FP"
    assert trend_row["p_value_corrected"] < 0.01

    # vol_z is noise: should NOT be significant
    vol_row = summary[summary["feature"] == "vol_z"].iloc[0]
    assert vol_row["p_value_corrected"] > 0.05


def test_insight_handles_empty_groups():
    nb = pd.DataFrame({"mistake_type": ["TP", "TN"], "rsi": [50, 51]})
    out = generate_failure_insights(nb, failure_class="FP")
    assert out["summary"].empty
    assert "insufficient" in out["report"]


def test_insight_report_contains_sections():
    nb = _planted_notebook()
    out = generate_failure_insights(nb, failure_class="FP")
    assert "Insight Report" in out["report"]
    assert "Significantly elevated" in out["report"]
