"""
Re-analyze failures from an already-trained model + OOS predictions.

Usage:
    python scripts/analyze_failures.py path/to/ohlcv.csv \
        --model model.json --oof oof.csv \
        --notebook-out notebook.csv --cluster 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fibtrader import BotConfig, add_features, attach_recent_swing_and_fib, zigzag_confirmed
from fibtrader.data import load_ohlcv_csv
from fibtrader.ml import (
    build_dataset,
    build_failure_notebook,
    cluster_failures,
    triple_barrier_labels,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--zigzag-pct", type=float, default=0.05)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--notebook-out", required=True)
    ap.add_argument("--cluster", type=int, default=0)
    args = ap.parse_args()

    import xgboost as xgb

    cfg = BotConfig(
        zigzag_pct=args.zigzag_pct,
        hold_bars=args.hold_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    df = load_ohlcv_csv(args.csv)
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    feat = add_features(sw, cfg)
    feat = triple_barrier_labels(feat, cfg)
    X, y, _ = build_dataset(feat)

    oof_csv = pd.read_csv(args.oof, index_col=0)
    if "proba" in oof_csv.columns:
        oof_proba = oof_csv["proba"]
    else:
        oof_proba = oof_csv.iloc[:, 0]

    model = xgb.XGBClassifier()
    model.load_model(args.model)

    nb = build_failure_notebook(
        feat, oof_proba, model, X,
        threshold=args.threshold, top_k_shap=5, only_failures=False,
    )
    if args.cluster > 0:
        nb = cluster_failures(nb, n_clusters=args.cluster, mistake_type="FP")
    nb.to_csv(args.notebook_out)
    print(f"Wrote -> {args.notebook_out}")
    print(nb.groupby("mistake_type").size().to_dict())


if __name__ == "__main__":
    main()
