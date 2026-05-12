"""CLI: run a single backtest on a CSV of OHLCV bars."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibtrader import (
    BotConfig,
    add_features,
    attach_recent_swing_and_fib,
    backtest_fib_confluence,
    zigzag_confirmed,
)
from fibtrader.data import load_ohlcv_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="OHLCV CSV path")
    ap.add_argument("--zigzag-pct", type=float, default=0.05)
    ap.add_argument("--min-conf", type=int, default=3)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    cfg = BotConfig(
        zigzag_pct=args.zigzag_pct,
        min_confluence=args.min_conf,
        hold_bars=args.hold_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    df = load_ohlcv_csv(args.csv)
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    feat = add_features(sw, cfg)
    trades, stats = backtest_fib_confluence(feat, cfg)

    print(json.dumps(stats, indent=2, default=str))
    if args.trades_out and trades is not None and not trades.empty:
        trades.to_csv(args.trades_out, index=False)


if __name__ == "__main__":
    main()
