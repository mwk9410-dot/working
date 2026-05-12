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

from fibtrader import BotConfig, prepare_features
from fibtrader.altdata import MassiveCSVAdapter, merge_altdata
from fibtrader.data import load_ohlcv_csv
from fibtrader.ml import (
    build_dataset,
    build_failure_notebook,
    cluster_failures,
    generate_failure_insights,
    shap_importance,
    train_xgb_rolling,
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
    ap.add_argument("--rolling", action="store_true",
                    help="bar-by-bar rolling walk-forward instead of K-fold")
    ap.add_argument("--rolling-min-train", type=int, default=252)
    ap.add_argument("--rolling-step", type=int, default=1)
    ap.add_argument("--refit-every", type=int, default=5,
                    help="retrain interval in rolling mode (1 = strict daily)")
    ap.add_argument("--notebook-out", default=None,
                    help="path to save the per-row failure notebook CSV")
    ap.add_argument("--cluster-failures", type=int, default=0,
                    help="cluster FP rows into K groups (0 = off)")
    ap.add_argument("--insight-out", default=None,
                    help="path to save the failure-insight text report")
    ap.add_argument("--insight-class", default="FP",
                    help="mistake class to analyze (FP, FN)")
    ap.add_argument("--max-train-size", type=int, default=None,
                    help="sliding window 크기 (None=누적식, 정수=고정식). "
                         "대규모 학습이면 504~1260 권장")
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

    feat = prepare_features(df, cfg)
    feat = triple_barrier_labels(feat, cfg)

    X, y, idx = build_dataset(feat)
    print(f"Dataset: X={X.shape}, positives={int(y.sum())}/{len(y)} ({y.mean():.2%})")

    if len(X) < 200:
        print("Not enough labeled rows; aborting.")
        return

    embargo = args.embargo if args.embargo is not None else cfg.hold_bars
    if args.rolling:
        print(f"Rolling mode: min_train={args.rolling_min_train}, step={args.rolling_step}, "
              f"refit_every={args.refit_every}, embargo={embargo}")
        result = train_xgb_rolling(
            X, y,
            min_train=args.rolling_min_train,
            embargo=embargo,
            step=args.rolling_step,
            refit_every=args.refit_every,
            threshold=args.threshold,
            verbose=True,
        )
        print(f"Rolling OOS rows: {int(result.aggregate_metrics.get('n_oof', 0))}")
        print("Aggregate OOS:", json.dumps(result.aggregate_metrics, default=str))
    else:
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

    notebook_df = None
    if args.notebook_out:
        nb = build_failure_notebook(
            feat,
            result.oof_proba,
            result.final_model,
            X,
            threshold=args.threshold,
            top_k_shap=5,
            only_failures=False,
        )
        if args.cluster_failures > 0:
            nb = cluster_failures(nb, n_clusters=args.cluster_failures, mistake_type="FP")
        nb.to_csv(args.notebook_out)
        print(f"Wrote failure notebook -> {args.notebook_out}")
        summary = nb.groupby("mistake_type").size().to_dict()
        print(f"Notebook summary: {summary}")
        fp = nb[nb["mistake_type"] == "FP"]
        if not fp.empty:
            avg_fp_ret = fp["forward_return"].mean()
            print(f"FP count={len(fp)}, mean forward_return={avg_fp_ret:+.4f}")
            if "cluster_id" in fp.columns and fp["cluster_id"].notna().any():
                cluster_summary = (
                    fp.groupby("cluster_id")["forward_return"]
                    .agg(["count", "mean", "min"])
                    .round(4)
                )
                print("FP clusters (by forward_return):")
                print(cluster_summary)
        notebook_df = nb

    if args.insight_out and notebook_df is not None:
        insight = generate_failure_insights(
            notebook_df, failure_class=args.insight_class, top_k=15
        )
        with open(args.insight_out, "w") as fh:
            fh.write(insight["report"])
        summary_csv = args.insight_out.rsplit(".", 1)[0] + "_summary.csv"
        insight["summary"].to_csv(summary_csv, index=False)
        print(f"Wrote insight report -> {args.insight_out}")
        print(f"Wrote insight summary table -> {summary_csv}")
        print(insight["report"])


if __name__ == "__main__":
    main()
