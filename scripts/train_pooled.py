"""
풀드(pooled) 학습 CLI — 여러 종목을 합쳐 한 모델 학습.

입력 (둘 중 하나):
    --dir   DIR    {ticker}.csv 파일들이 있는 디렉토리
    --long  PATH   'Ticker' 컬럼이 있는 long format CSV

권장 셋업 (대규모 데이터):
    python scripts/train_pooled.py --dir corpus/ \\
        --max-train-size 504 --n-splits 5 \\
        --min-conf 3 --hold-bars 20 \\
        --fee-bps 25 --slippage-bps 5 \\
        --pooled-out pooled.parquet \\
        --model-out model.json --oof-out oof.csv \\
        --shap-out shap.csv --notebook-out notebook.csv \\
        --insight-out insight.txt

  max-train-size 504 = sliding window 약 2년
  n-splits 5         = 검증 5회 (각 검증 구간 약 6개월~1년)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fibtrader import BotConfig
from fibtrader.ml import (
    build_dataset,
    build_failure_notebook,
    build_pooled_frame,
    cluster_failures,
    generate_failure_insights,
    shap_importance,
    train_xgb_walkforward,
)
from fibtrader.paths import artifacts_dir, corpus_dir
from fibtrader.universe import load_corpus_dir, load_corpus_long


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dir", help="{ticker}.csv 들이 든 디렉토리. "
                                  "미지정 시 FIBTRADER_CORPUS_DIR 환경변수 사용")
    g.add_argument("--long", help="'Ticker' 컬럼이 있는 long format CSV")

    ap.add_argument("--max-tickers", type=int, default=None,
                    help="디버깅용 — 처음 N 종목만 사용")

    # 전략 파라미터
    ap.add_argument("--zigzag-pct", type=float, default=0.05)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--take-atr", type=float, default=2.5)
    ap.add_argument("--min-conf", type=int, default=3)
    ap.add_argument("--fee-bps", type=float, default=25.0,
                    help="한국 리테일 미국 주식 약 25bps/측")
    ap.add_argument("--slippage-bps", type=float, default=5.0)

    # 학습 파라미터
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=None,
                    help="기본은 hold_bars 값으로")
    ap.add_argument("--max-train-size", type=int, default=504,
                    help="sliding window 크기 (None = 누적식). 권장 504 (약 2년)")
    ap.add_argument("--min-train", type=int, default=2520,
                    help="첫 검증 fold 전에 최소 누적할 학습 행 수")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=4)

    # 출력 — 기본은 FIBTRADER_ARTIFACTS_DIR (없으면 ./data/artifacts)
    _art = artifacts_dir()
    ap.add_argument("--pooled-out", default=None,
                    help="합쳐진 feature+label DataFrame parquet 저장 (선택)")
    ap.add_argument("--model-out", default=str(_art / "pooled_model.json"))
    ap.add_argument("--oof-out", default=str(_art / "pooled_oof.csv"))
    ap.add_argument("--shap-out", default=str(_art / "pooled_shap.csv"))
    ap.add_argument("--importance-out", default=str(_art / "pooled_importance.csv"))
    ap.add_argument("--notebook-out", default=None)
    ap.add_argument("--insight-out", default=None)
    ap.add_argument("--cluster-failures", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="풀드 빌드 병렬 워커 수. 6500종목이면 8~16 권장")

    args = ap.parse_args()

    print("=== 1) 종목 corpus 로딩 ===")
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
        stop_atr_mult=args.stop_atr,
        take_atr_mult=args.take_atr,
        min_confluence=args.min_conf,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    print(f"\n=== 2) 모든 종목 파이프라인 → 풀드 frame 생성 (workers={args.workers}) ===")
    pooled = build_pooled_frame(corpus, cfg, n_workers=args.workers, verbose=True)
    if pooled.empty:
        print("풀드 결과가 비어 있음. 종료.")
        return
    print(f"풀드 행 수: {len(pooled):,}, 라벨된 행: {int(pooled['label'].notna().sum()):,}")
    print(f"기간: {pooled['Date'].min()} ~ {pooled['Date'].max()}")

    if args.pooled_out:
        pooled.to_parquet(args.pooled_out, index=False)
        print(f"풀드 frame 저장 -> {args.pooled_out}")

    print("\n=== 3) 학습 데이터셋 생성 ===")
    X, y, idx = build_dataset(pooled)
    print(f"X={X.shape}, 양성={int(y.sum()):,}/{len(y):,} ({y.mean():.2%})")
    if len(X) < args.min_train + 100:
        print(f"학습 데이터가 너무 적음 (필요: {args.min_train + 100})")
        return

    print("\n=== 4) Sliding window walk-forward 학습 ===")
    embargo = args.embargo if args.embargo is not None else cfg.hold_bars
    print(f"  fold 수={args.n_splits}, sliding 크기={args.max_train_size}, "
          f"embargo={embargo}, 최소 학습 행={args.min_train}")
    result = train_xgb_walkforward(
        X, y,
        n_splits=args.n_splits,
        embargo=embargo,
        min_train=args.min_train,
        max_train_size=args.max_train_size,
        xgb_params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
        threshold=args.threshold,
    )

    print("\nfold 별 성과:")
    for i, m in enumerate(result.fold_metrics):
        print(f"  fold {i}: {json.dumps(m, default=str)}")
    print(f"\n전체 OOS: {json.dumps(result.aggregate_metrics, default=str, indent=2)}")

    print("\n=== 5) 저장 ===")
    result.feature_importance.to_csv(args.importance_out, header=["gain_importance"])
    print(f"  중요도 -> {args.importance_out}")

    try:
        shap_df = shap_importance(result.final_model, X)
        shap_df.to_csv(args.shap_out)
        print(f"  SHAP -> {args.shap_out}")
        print("\nSHAP 상위 15개:")
        print(shap_df.head(15))
    except Exception as e:
        print(f"  SHAP 실패: {e}")

    oof_df = pd.DataFrame({
        "proba": result.oof_proba,
        "label": y.reindex(result.oof_proba.index),
        "Date": pooled.loc[result.oof_proba.index, "Date"].values,
        "Ticker": pooled.loc[result.oof_proba.index, "Ticker"].values,
    })
    oof_df.to_csv(args.oof_out)
    print(f"  OOS 예측 -> {args.oof_out}")

    result.final_model.save_model(args.model_out)
    print(f"  최종 모델 (전체 데이터로 학습) -> {args.model_out}")

    if args.notebook_out:
        print("\n=== 6) 오답노트 ===")
        nb = build_failure_notebook(
            pooled, result.oof_proba, result.final_model, X,
            threshold=args.threshold, top_k_shap=5, only_failures=False,
        )
        # 풀드용 식별자 추가
        if "Ticker" not in nb.columns and "Ticker" in pooled.columns:
            nb["Ticker"] = pooled.loc[nb.index, "Ticker"].values
        if args.cluster_failures > 0:
            nb = cluster_failures(nb, n_clusters=args.cluster_failures, mistake_type="FP")
        nb.to_csv(args.notebook_out)
        print(f"  오답노트 -> {args.notebook_out}")
        summary = nb.groupby("mistake_type").size().to_dict()
        print(f"  요약: {summary}")

        if args.insight_out:
            print("\n=== 7) 인사이트 리포트 ===")
            insight = generate_failure_insights(nb, failure_class="FP", top_k=15)
            with open(args.insight_out, "w") as fh:
                fh.write(insight["report"])
            print(f"  -> {args.insight_out}")
            print(insight["report"])


if __name__ == "__main__":
    main()
