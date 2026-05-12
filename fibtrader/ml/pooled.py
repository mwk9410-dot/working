"""
풀드(pooled) 학습용 데이터셋 빌더.

여러 종목의 OHLCV → 각각 swing/fib/features/label 파이프라인을 돌린 뒤
모두 합쳐 하나의 큰 DataFrame을 만든다. walk-forward 검증이 의미를 가지려면
반드시 Date 기준으로 정렬해야 한다 — 그래야 학습 구간이 검증 구간보다
시간적으로 앞선다는 보장이 생긴다.

각 행에는 출처 'Ticker' 컬럼이 붙어 사후 분석(어떤 종목이 손실에 기여했는지)
이 가능하다.

종목별 파이프라인은 독립이므로 multiprocessing 으로 병렬화. 6500종목 같은
대규모 corpus 에서 wall-clock 시간이 코어 수에 비례해 줄어든다.
"""
from __future__ import annotations

from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import BotConfig
from ..pipeline import prepare_features
from .labels import triple_barrier_labels


def _process_one(args: Tuple[str, pd.DataFrame, BotConfig]) -> Optional[pd.DataFrame]:
    ticker, df, cfg = args
    if df is None or df.empty or "Date" not in df.columns:
        return None
    try:
        feat = prepare_features(df, cfg)
        feat = triple_barrier_labels(feat, cfg)
        feat["Ticker"] = ticker
        return feat
    except Exception:
        return None


def build_pooled_frame(
    corpus: Dict[str, pd.DataFrame],
    cfg: BotConfig,
    n_workers: int = 1,
    downcast_float: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    corpus: ticker -> OHLCV DataFrame (Date 컬럼 필수)
    n_workers: 1 이면 직렬, 2 이상이면 multiprocessing.Pool
    downcast_float: True 면 합친 뒤 float64 → float32 다운캐스트 (메모리 절반)
    반환: 모든 종목을 시간순으로 정렬한 단일 feature+label DataFrame.
          'Ticker' 컬럼이 추가된다.
    """
    pool_input = [(tk, df, cfg) for tk, df in corpus.items()]
    if n_workers <= 1:
        results = [_process_one(args) for args in pool_input]
    else:
        with Pool(n_workers) as p:
            results = p.map(_process_one, pool_input)

    frames = [r for r in results if r is not None and not r.empty]
    skipped = len(pool_input) - len(frames)

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    # walk-forward 가 시간 순서를 따라가도록 정렬 (같은 날짜는 ticker 기준 안정 정렬)
    pooled = pooled.sort_values(["Date", "Ticker"], kind="mergesort").reset_index(drop=True)

    if downcast_float:
        float_cols = pooled.select_dtypes(include=["float64"]).columns
        pooled[float_cols] = pooled[float_cols].astype(np.float32)

    if verbose:
        mem_mb = pooled.memory_usage(deep=True).sum() / 1024**2
        print(f"풀드 결과: 종목 {len(frames)}개, 행 {len(pooled):,}개, "
              f"라벨 {int(pooled['label'].notna().sum()):,}개, "
              f"스킵 {skipped}, 메모리 {mem_mb:.1f}MB")
    return pooled
