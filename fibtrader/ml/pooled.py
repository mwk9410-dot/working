"""
풀드(pooled) 학습용 데이터셋 빌더.

여러 종목의 OHLCV → 각각 swing/fib/features/label 파이프라인을 돌린 뒤
모두 합쳐 하나의 큰 DataFrame을 만든다. walk-forward 검증이 의미를 가지려면
반드시 Date 기준으로 정렬해야 한다 — 그래야 학습 구간이 검증 구간보다
시간적으로 앞선다는 보장이 생긴다.

각 행에는 출처 'Ticker' 컬럼이 붙어 사후 분석(어떤 종목이 손실에 기여했는지)
이 가능하다.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..config import BotConfig
from ..features import add_features
from ..fib import attach_recent_swing_and_fib
from ..swing import zigzag_confirmed
from .labels import triple_barrier_labels


def build_pooled_frame(
    corpus: Dict[str, pd.DataFrame],
    cfg: BotConfig,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    corpus: ticker -> OHLCV DataFrame (Date 컬럼 필수)
    반환: 모든 종목을 시간순으로 정렬한 단일 feature+label DataFrame.
          'Ticker' 컬럼이 추가된다.
    """
    frames = []
    skipped = 0
    for ticker, df in corpus.items():
        if df is None or df.empty or "Date" not in df.columns:
            skipped += 1
            continue
        try:
            sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
            sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
            feat = add_features(sw, cfg)
            feat = triple_barrier_labels(feat, cfg)
            feat["Ticker"] = ticker
            frames.append(feat)
            if verbose:
                print(f"  {ticker}: {len(feat)} rows, labeled={feat['label'].notna().sum()}")
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"  {ticker}: skipped ({e})")

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    # walk-forward 가 시간 순서를 따라가도록 정렬 (같은 날짜는 ticker 기준 안정 정렬)
    pooled = pooled.sort_values(["Date", "Ticker"], kind="mergesort").reset_index(drop=True)
    if verbose:
        print(f"풀드 결과: 종목 {len(frames)}개, 행 {len(pooled):,}개, "
              f"라벨 {int(pooled['label'].notna().sum()):,}개, "
              f"스킵 {skipped}")
    return pooled
