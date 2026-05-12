"""
Propose entry-blocking rules from a failure notebook.

The output JSON is a list of CandidateRule objects with enabled=False. The
user must manually flip `enabled` to true for the rules they accept.

Usage:
    python scripts/propose_rules.py notebook.csv \\
        --max-depth 2 --min-fp 5 --min-ratio 2.0 \\
        --out rules_candidates.json

After acceptance, the same JSON can be loaded by your live signal layer:

    from fibtrader.ml import load_rules, apply_rules
    rules = load_rules("rules_v1.json")
    blocked = apply_rules(feat_df, rules)         # only enabled rules count
    entries = signal_mask & ~blocked
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fibtrader.ml import propose_rules, save_rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook_csv")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-fp", type=int, default=5,
                    help="minimum FPs the rule must block")
    ap.add_argument("--min-ratio", type=float, default=2.0,
                    help="minimum FP-blocked / TP-blocked ratio")
    ap.add_argument("--purity", type=float, default=0.7,
                    help="required FP share of blocked rows")
    ap.add_argument("--out", default="rules_candidates.json")
    args = ap.parse_args()

    nb = pd.read_csv(args.notebook_csv, index_col=0)
    rules = propose_rules(
        nb,
        max_depth=args.max_depth,
        min_fp_blocked=args.min_fp,
        min_fp_tp_ratio=args.min_ratio,
        target_fp_purity=args.purity,
    )

    if not rules:
        print("No rules met the thresholds. Try relaxing --min-ratio or --purity.")
        return

    save_rules(rules, args.out)
    print(f"Proposed {len(rules)} rules -> {args.out}")
    print(f"All rules are saved with enabled=false. Flip enabled=true to activate.\n")
    for r in rules:
        print(f"  [{r.name}] {r.predicate}")
        print(f"    blocked FP={r.n_blocked_fp}, TP={r.n_blocked_tp}, ratio={r.fp_tp_ratio:.2f}, "
              f"avg_ret_blocked={r.blocked_avg_return:+.4f}")


if __name__ == "__main__":
    main()
