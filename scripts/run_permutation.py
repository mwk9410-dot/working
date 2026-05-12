"""CLI: permutation test of Fib ratios vs random ratios."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibtrader import BotConfig, permutation_test_random_ratios, zigzag_confirmed
from fibtrader.data import load_ohlcv_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n-perms", type=int, default=500)
    ap.add_argument("--metric", default="avg_ret", choices=["avg_ret", "win_rate", "sharpe_like", "cum_return"])
    args = ap.parse_args()

    cfg = BotConfig(n_permutations=args.n_perms)
    df = load_ohlcv_csv(args.csv)
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    result = permutation_test_random_ratios(sw, cfg, metric=args.metric)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
