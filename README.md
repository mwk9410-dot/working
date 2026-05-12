# fibtrader

피보나치 되돌림 + 다중지표 confluence 기반 롱 트레이딩 봇. 학술적 검증 우선 설계:
look-ahead bias 제거, 거래비용 반영, 무작위 비율 대비 permutation test 내장.

## 설계 핵심

- **Confirmed ZigZag**: pivot은 일정 비율 반대 움직임 + `confirm_bars` 이후에만 사용 가능.
- **Confluence gating**: `fib_near` + `trend_up` (MA20>MA50>MA200) + RSI/MACD/BB/Volume 중
  `min_confluence` 이상일 때만 진입. Fib 단독 진입은 금지.
- **Regime filter**: ATR%가 `[atr_pct_min, atr_pct_max]` 범위일 때만 활성.
- **Cost-aware backtest**: 왕복 fee + slippage, intrabar에서 stop이 take보다 우선 (보수적).
- **Permutation test**: 같은 스윙에 대해 Fib 5 비율 vs 무작위 5 비율 성과 차이의 p-value.
- **Walk-forward**: `walk_forward_backtest`로 OOS fold별 평가.

## 디렉토리

```
fibtrader/         indicators, swing, fib, features, backtest, stats, signal, config, data
scripts/           run_backtest.py, run_permutation.py
tests/             pytest 기반 단위 테스트 (합성 OHLCV 사용)
```

## 빠른 실행

```bash
pip install -r requirements.txt
python scripts/run_backtest.py path/to/ohlcv.csv --zigzag-pct 0.05 --min-conf 3
python scripts/run_permutation.py path/to/ohlcv.csv --n-perms 500 --metric avg_ret
pytest -q
```

## 1분봉 / 소형주에서 쓸 때 주의

본 봇의 1분봉 적용 가능성은 제한적입니다. `BotConfig`에서 반드시 다음을 조정하세요.

- `annualization_factor`: 분봉이면 `252 * 390` 정도.
- `fee_bps`, `slippage_bps`: 소형주는 보수적으로 20bps+.
- `atr_pct_min/max`: 변동성 regime 필터 재조정.
- 외부에서 거래대금/스프레드/호가 imbalance 필터를 사전 적용한 후 `signal.LiveSignalGenerator`에 공급.

## 라이브 모드 골격

```python
from fibtrader import BotConfig, LiveSignalGenerator
gen = LiveSignalGenerator(BotConfig())
gen.warmup(history_df)             # 최소 ~250 bars
decision = gen.on_new_bar(latest_bar)
if decision and decision.action == "enter_long":
    broker.submit(decision)
```

## 통계적 검증

`scripts/run_permutation.py`는 동일 스윙·동일 confluence 룰에서 Fib 5 비율을
[0.2, 0.8] 균등 분포 무작위 5 비율로 바꿔 재실행, 양측 empirical p-value와
Cohen's d 효과크기를 출력합니다. 거래비용 후에도 baseline이 perm 분포의
상위 5% 안에 들어와야 비로소 "비율의 특수성" 가설을 약하게 지지할 수
있습니다.
