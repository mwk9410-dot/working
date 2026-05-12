"""
유니버스 분포 / 생존편향 진단 스크립트.

로컬 분봉 데이터 폴더를 받아 다음을 출력:
  1. 종목 개수, 총 용량, 파일 크기 분포
  2. 파일명 패턴 (티커 형식, 의심 접미사)
  3. 종목별 데이터 기간 분포 — 모두 같으면 생존편향 의심,
     종목마다 다르면 PIT (Point-in-Time) 가능성
  4. 종목별 봉 개수 분포 (상폐로 일찍 끝난 종목 비율)
  5. 거래대금 분포 (전체 기간 평균)

사용법:
    python scripts/inspect_corpus.py "C:\\Users\\mwk94\\mw94\\data\\cache\\1m_massive"

또는 출력을 파일로:
    python scripts/inspect_corpus.py "C:\\...\\1m_massive" --out report.txt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from fibtrader.paths import corpus_dir


def _peek_csv(path: Path, n_head: int = 2, n_tail: int = 2):
    """전체 안 읽고 첫·끝 몇 줄, 행 수만 추정."""
    try:
        head = pd.read_csv(path, nrows=n_head)
    except Exception:
        return None
    # 전체 행 수 (헤더 제외) 빠르게
    with open(path, "rb") as f:
        n_rows = sum(1 for _ in f) - 1
    # 마지막 n_tail 행
    try:
        tail = pd.read_csv(path, skiprows=range(1, max(1, n_rows - n_tail + 1)))
    except Exception:
        tail = head
    return head, tail, n_rows


def _find_datetime_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "datetime", "time", "timestamp"):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default=None,
                    help="분봉 CSV 폴더. 미지정 시 FIBTRADER_CORPUS_DIR 환경변수 사용")
    ap.add_argument("--sample", type=int, default=200,
                    help="기간 분포 분석에 사용할 표본 종목 수")
    ap.add_argument("--out", default=None, help="결과를 이 파일에 저장")
    args = ap.parse_args()

    d = Path(args.directory) if args.directory else corpus_dir()
    print(f"corpus 폴더: {d}")
    if not d.exists():
        print(f"폴더 없음: {d}")
        sys.exit(1)

    files = sorted([p for p in d.iterdir() if p.suffix.lower() in (".csv", ".parquet")])
    n = len(files)
    if n == 0:
        print(f"CSV/parquet 파일 없음")
        sys.exit(1)

    sizes = np.array([p.stat().st_size for p in files])
    total_gb = sizes.sum() / 1024**3

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 70)
    out(f"유니버스 분포 진단: {d}")
    out("=" * 70)

    # 1) 종목 수 / 파일 크기
    out(f"\n[1] 종목 수 = {n:,}개, 총 용량 = {total_gb:.2f} GB")
    out(f"    파일당 크기: min={sizes.min()/1024:.0f}KB, "
        f"median={np.median(sizes)/1024**2:.1f}MB, "
        f"max={sizes.max()/1024**2:.1f}MB")

    # 2) 파일명 패턴
    out("\n[2] 파일명 샘플 (앞·뒤 10개)")
    for p in files[:10]:
        out(f"    {p.name}")
    out("    ...")
    for p in files[-10:]:
        out(f"    {p.name}")

    suspicious = [p.name for p in files if any(
        s in p.name.lower() for s in ("_delisted", "_old", "_bk", "_dead", ".oq", ".k", "-old")
    )]
    out(f"\n    의심 접미사 (delisted/old 등) 가진 파일: {len(suspicious)}개")
    if suspicious[:5]:
        for s in suspicious[:5]:
            out(f"      {s}")

    # 3) 종목별 기간 분석 — 표본만
    sample_files = files if n <= args.sample else \
        list(np.random.RandomState(42).choice(files, args.sample, replace=False))

    rows = []
    out(f"\n[3] 기간 분석 (표본 {len(sample_files)}개 종목)")
    for p in sample_files:
        peek = _peek_csv(p)
        if peek is None:
            continue
        head, tail, n_rows = peek
        dt_col = _find_datetime_col(head)
        if dt_col is None:
            continue
        try:
            first = pd.to_datetime(head[dt_col].iloc[0])
            last = pd.to_datetime(tail[dt_col].iloc[-1])
        except Exception:
            continue
        rows.append({
            "ticker": p.stem,
            "first": first,
            "last": last,
            "n_rows": n_rows,
            "size_mb": p.stat().st_size / 1024**2,
        })

    if not rows:
        out("    날짜 컬럼을 찾을 수 없음")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    out(f"    첫 봉 날짜 분포:")
    out(f"      min={summary['first'].min()},  median={summary['first'].median()},  "
        f"max={summary['first'].max()}")
    out(f"    마지막 봉 날짜 분포:")
    out(f"      min={summary['last'].min()},  median={summary['last'].median()},  "
        f"max={summary['last'].max()}")

    # 생존편향 진단
    first_var_days = (summary["first"].max() - summary["first"].min()).days
    last_var_days = (summary["last"].max() - summary["last"].min()).days
    out(f"\n    첫 봉 날짜 변동폭: {first_var_days}일")
    out(f"    마지막 봉 날짜 변동폭: {last_var_days}일")

    if first_var_days < 30 and last_var_days < 30:
        verdict = "❗ 생존편향 가능성 큼 — 모든 종목이 거의 같은 기간"
    elif last_var_days > 365:
        verdict = "✓ 종목별 종료 시점이 다양 → 상폐 종목 포함 가능성 (PIT 기대)"
    else:
        verdict = "△ 애매 — 종목별 첫 봉은 다르지만 종료는 거의 같음"
    out(f"    진단: {verdict}")

    # 4) 봉 개수 분포
    out(f"\n[4] 봉 개수 분포 (표본 {len(summary)}개)")
    out(f"    min={summary['n_rows'].min():,}, "
        f"p25={summary['n_rows'].quantile(0.25):,.0f}, "
        f"median={summary['n_rows'].median():,.0f}, "
        f"p75={summary['n_rows'].quantile(0.75):,.0f}, "
        f"max={summary['n_rows'].max():,}")

    # 5) 행 수가 극히 적은 종목 (early 상폐 가능성)
    short = summary[summary["n_rows"] < summary["n_rows"].median() * 0.3]
    out(f"\n    중간값의 30% 미만 행만 있는 종목 (조기 상폐 후보): {len(short)}개")
    if len(short) > 0:
        out(f"      예: {', '.join(short['ticker'].head(10).tolist())}")

    # 6) 거래대금 표본 (첫 종목 하나만)
    if rows:
        try:
            sample = pd.read_csv(sample_files[0])
            cols = {c.lower(): c for c in sample.columns}
            if "close" in cols and "volume" in cols:
                close = sample[cols["close"]]
                vol = sample[cols["volume"]]
                dv = (close * vol).median()
                out(f"\n[5] 거래대금 예시 ({sample_files[0].stem}): "
                    f"median dollar volume = ${dv:,.0f}")
        except Exception:
            pass

    out("\n" + "=" * 70)
    out("결론:")
    out("=" * 70)
    out(f"  - 종목 수: {n:,}")
    out(f"  - 표본 기준 진단: {verdict}")
    if first_var_days >= 30:
        out("  → PIT universe (시점별 필터링) 구현 가능")
    if last_var_days < 30:
        out("  ⚠ 상폐 종목 데이터가 빠진 것 같음 — Massive.io에서 옵션 확인 필요")

    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"\n저장 -> {args.out}")


if __name__ == "__main__":
    main()
