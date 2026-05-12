"""
CLI: train an XGBoost classifier on Fib-confluence features with
embargoed walk-forward CV, dump SHAP importance, and save the model.

Usage:
    python scripts/train_xgb.py path/to/ohlcv.csv \
        --model-out model.json \
        --importance-out importance.csv \
        --altdata-csv massive_export.csv

Optional Massive.io alt-data: pass --altdata-csv with columns
    Date, Ticker, metric, value
(long format). Lag is handled internally.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fibtrader import BotConfig, add_features, attach_recent_swing_and_fib, zigzag_confirmed
from fibtrader.altdata import MassiveCSVAdapter, merge_altdata
from fibtrader.data import load_ohlcv_csv
from fibtrader.ml import (
    build_dataset,
    shap_importance,
    train_xgb_walkforward,
    triple_barrier_labels,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--zigzag-pct", type=float, default=0.05)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--take-atr", type=float, default=2.5)
    ap.add_argument("--fee-bps", type=float, default=25.0, help="Korean retail US-stock fee approx 25bps/side")
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=None, help="defaults to hold_bars")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model-out", default="model.json")
    ap.add_argument("--importance-out", default="importance.csv")
    ap.add_argument("--shap-out", default="shap.csv")
    ap.add_argument("--oof-out", default="oof.csv")
    ap.add_argument("--altdata-csv", default=None)
    ap.add_argument("--altdata-ticker", default=None)
    args = ap.parse_args()

    cfg = BotConfig(
        zigzag_pct=args.zigzag_pct,
        hold_bars=args.hold_bars,
        stop_atr_mult=args.stop_atr,
        take_atr_mult=args.take_atr,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    df = load_ohlcv_csv(args.csv)

    if args.altdata_csv:
        adapter = MassiveCSVAdapter(path=args.altdata_csv, ticker=args.altdata_ticker)
        alt = adapter.load()
        df = merge_altdata(df, alt, date_col="Date")
        print(f"Merged alt-data: {len([c for c in df.columns if c.startswith('alt_')])} columns")

    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    feat = add_features(sw, cfg)
    feat = triple_barrier_labels(feat, cfg)

    X, y, idx = build_dataset(feat)
    print(f"Dataset: X={X.shape}, positives={int(y.sum())}/{len(y)} ({y.mean():.2%})")

    if len(X) < 200:
        print("Not enough labeled rows; aborting.")
        return

    embargo = args.embargo if args.embargo is not None else cfg.hold_bars
    result = train_xgb_walkforward(
        X, y,
        n_splits=args.n_splits,
        embargo=embargo,
        threshold=args.threshold,
    )

    print("Per-fold metrics:")
    for i, m in enumerate(result.fold_metrics):
        print(f"  fold {i}: {json.dumps(m, default=str)}")
    print("Aggregate OOS:", json.dumps(result.aggregate_metrics, default=str))

    result.feature_importance.to_csv(args.importance_out, header=["gain_importance"])
    print(f"Wrote gain importance -> {args.importance_out}")

    try:
        shap_df = shap_importance(result.final_model, X)
        shap_df.to_csv(args.shap_out)
        print(f"Wrote SHAP importance -> {args.shap_out}")
        print("Top 15 by SHAP:")
        print(shap_df.head(15))
    except Exception as e:
        print(f"SHAP failed: {e}")

    oof_df = pd.DataFrame({"proba": result.oof_proba, "label": y.reindex(result.oof_proba.index)})
    oof_df.to_csv(args.oof_out)
    print(f"Wrote OOS probabilities -> {args.oof_out}")

    result.final_model.save_model(args.model_out)
    print(f"Saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
