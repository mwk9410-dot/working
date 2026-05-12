"""
Universe sensitivity sweep CLI.

Inputs (one of):
    --dir DIR        a directory containing {ticker}.csv files
    --long PATH      a single long-format CSV with a 'Ticker' column

Output:
    sweep.csv     long-format: dimension, threshold, n_tickers, mean, median, ...
    backtest.csv  per-ticker backtest stats (cached so repeated sweeps are fast)
    stats.csv     per-ticker liquidity stats
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fibtrader import BotConfig
from fibtrader.paths import artifacts_dir, corpus_dir
from fibtrader.universe import (
    FilterSpec,
    backtest_corpus,
    compute_ticker_stats,
    load_corpus_dir,
    load_corpus_long,
    sensitivity_sweep,
)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dir", help="{ticker}.csv 폴더. 미지정 시 FIBTRADER_CORPUS_DIR 환경변수 사용")
    g.add_argument("--long", help="'Ticker' 컬럼이 있는 long format CSV")

    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--zigzag-pct", type=float, default=0.05)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--min-conf", type=int, default=3)
    ap.add_argument("--metric", default="sharpe_like",
                    choices=["sharpe_like", "avg_ret", "cum_return", "win_rate", "profit_factor"])
    ap.add_argument("--workers", type=int, default=1)
    _art = artifacts_dir()
    ap.add_argument("--sweep-out", default=str(_art / "sweep.csv"))
    ap.add_argument("--backtest-out", default=str(_art / "backtest.csv"))
    ap.add_argument("--stats-out", default=str(_art / "stats.csv"))
    args = ap.parse_args()

    print("corpus 로딩...")
    if args.long:
        corpus = load_corpus_long(args.long)
        if args.max_tickers is not None:
            keys = list(corpus.keys())[: args.max_tickers]
            corpus = {k: corpus[k] for k in keys}
    else:
        src = Path(args.dir) if args.dir else corpus_dir()
        print(f"  corpus 폴더: {src}")
        corpus = load_corpus_dir(src, max_tickers=args.max_tickers)
    print(f"로드된 종목: {len(corpus)}개")

    cfg = BotConfig(
        zigzag_pct=args.zigzag_pct,
        hold_bars=args.hold_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        min_confluence=args.min_conf,
    )

    print("Computing ticker stats...")
    stats = compute_ticker_stats(corpus)
    stats.to_csv(args.stats_out)
    print(f"Wrote -> {args.stats_out}")

    print(f"Running backtests on {len(corpus)} tickers (workers={args.workers})...")
    bt = backtest_corpus(corpus, cfg, n_workers=args.workers)
    bt.to_csv(args.backtest_out, index=False)
    print(f"Wrote -> {args.backtest_out}")
    if not bt.empty:
        print(f"Tickers producing trades: {len(bt)} / {len(corpus)}")
        print(f"Overall {args.metric}: mean={bt[args.metric].mean():.3f}, "
              f"median={bt[args.metric].median():.3f}")

    print("Sweeping filter dimensions...")
    sweep = sensitivity_sweep(
        corpus, cfg,
        per_ticker=bt,
        ticker_stats=stats,
        metric=args.metric,
    )
    sweep.to_csv(args.sweep_out, index=False)
    print(f"Wrote -> {args.sweep_out}")

    print("\n=== Sweep results ===")
    for dim, grp in sweep.groupby("dimension"):
        print(f"\n[{dim}]")
        print(grp[["threshold", "n_tickers", "mean", "median", "share_positive"]].to_string(index=False))


if __name__ == "__main__":
    main()
