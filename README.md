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

## ML 학습 (XGBoost + SHAP)

라벨은 백테스트 메커니즘과 동일한 **triple-barrier** (López de Prado).
CV는 **embargoed walk-forward** — overlapping forward window 누수 차단.

```bash
python scripts/train_xgb.py path/to/ohlcv.csv \
    --hold-bars 20 --stop-atr 1.5 --take-atr 2.5 \
    --fee-bps 25 --slippage-bps 5 \
    --model-out model.json --shap-out shap.csv --oof-out oof.csv
```

기본 비용은 한국 리테일의 미국주식 매매 (∼25bps/측)에 맞춰져 있습니다.
미국 리테일 $0 commission 환경이면 `--fee-bps 0`.

### Rolling daily walk-forward (D 학습 → D+gap 검증)

매 bar마다 학습-예측을 전진시키는 모드. `--refit-every`로 재학습 비용을 조절
하되, 예측은 매 bar 생성. `embargo`는 보통 `hold_bars` 이상.

```bash
python scripts/train_xgb.py ohlcv.csv --rolling \
    --rolling-min-train 252 --rolling-step 1 --refit-every 5 \
    --notebook-out notebook.csv --cluster-failures 5
```

- `step=1` + `refit-every=1`: 진성 daily refit (느림)
- `step=1` + `refit-every=5`: 주 1회 retrain, 매일 예측 (권장)
- `step=5` + `refit-every=20`: 빠른 1차 검증용

### 오답노트 (failure notebook)

`--notebook-out`을 켜면 OOS 결정마다 다음을 기록한 CSV가 생성됩니다.

| 컬럼 | 의미 |
|---|---|
| `mistake_type` | TP / FP / FN / TN |
| `oof_proba`, `oof_pred`, `label` | 예측 확률·결정·실제 |
| `forward_return`, `barrier_touched` | 실제 결과 (stop/take/time) |
| `top_pos_features` | 그 행에서 확률을 ↑ 시킨 SHAP 상위 5개 |
| `top_neg_features` | 그 행에서 확률을 ↓ 시킨 SHAP 상위 5개 |
| `rsi`, `atr_pct`, `trend_up`, `confluence_count`, `nearest_fib_level` 등 | 시장 context |
| `cluster_id` | (`--cluster-failures K` 시) FP를 K개 그룹으로 KMeans 클러스터링 |

이미 학습된 모델/OOS가 있으면 다시 학습하지 않고 분석만:

```bash
python scripts/analyze_failures.py ohlcv.csv \
    --model model.json --oof oof.csv \
    --notebook-out notebook.csv --cluster 5
```

### 인사이트 자동 추출

```bash
python scripts/train_xgb.py ohlcv.csv --rolling \
    --notebook-out notebook.csv --insight-out insight.txt
```

`generate_failure_insights`가 FP vs TP/TN을 각 feature별로 비교 (Welch's t /
two-proportion z / chi-squared), Holm-Bonferroni 보정 후 텍스트 리포트 생성.

### 규칙 제안기 (human-in-the-loop)

오답노트에서 단순한 진입 차단 규칙을 자동 제안 (DecisionTree depth≤2). 모든
후보는 `enabled=false`로 저장 — 사람이 명시적으로 켜기 전까지 적용 안 됨.

```bash
python scripts/propose_rules.py notebook.csv --max-depth 2 --min-fp 5 \
    --min-ratio 2.0 --out rules_candidates.json
# 검토 후 enabled: true 로 수정
```

규칙 활용:
```python
from fibtrader.ml import load_rules, apply_rules
rules = load_rules("rules_v1.json")
blocked = apply_rules(feat_df, rules)
entries = signal_mask & ~blocked
```

규칙 자동 제거는 `update_registry` + `recommend_removals` — 누적 FP/TP 비율이
낮아지거나 연속해서 안 걸리면 retire 후보로 플래그.

### Universe sensitivity sweep

```bash
python scripts/universe_sweep.py --dir /path/to/csv_dir/ \
    --metric sharpe_like --workers 8 \
    --sweep-out sweep.csv --backtest-out bt.csv
```

필터 한 dimension씩 변화 → `(threshold, n_tickers, 성과 분포)` 곡선 출력.
전체 grid 동시 최적화는 의도적으로 제외 (meta-overfitting 위험).

### Massive.io alt-data 통합

```bash
python scripts/train_xgb.py ohlcv.csv \
    --altdata-csv massive_export.csv --altdata-ticker AAPL
```

`MassiveCSVAdapter`는 long format (`Date,Ticker,metric,value`) 또는 wide
format을 수용하고, 출판 지연 (`release_lag_bdays`, 기본 1영업일)을 자동 보정한 후
`alt_*` prefix로 feature를 추가합니다.

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
