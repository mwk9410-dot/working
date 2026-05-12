"""
프로젝트 경로 헬퍼.

코드는 리포에, 데이터는 사용자 머신의 별도 위치에 둔다. 머신마다 경로가
다르므로 환경 변수로 주입한다.

우선순위:
  1. 함수 호출 시 인자로 명시적 전달
  2. 환경 변수 (FIBTRADER_*)
  3. 폴백: 리포 옆 ./data/ 폴더 (gitignore 처리됨)

.env 파일을 리포 루트에 두면 `python-dotenv` 가 자동 로드 (있을 때만).
없어도 OS 환경 변수면 됨.
"""
from __future__ import annotations

import os
from pathlib import Path

# .env 가 있고 python-dotenv 가 설치돼 있으면 한 번 로드
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


def _resolve(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    if val:
        return Path(val).expanduser().resolve()
    return default


def data_dir() -> Path:
    """전체 데이터 루트. FIBTRADER_DATA_DIR 또는 ./data/"""
    return _resolve(
        "FIBTRADER_DATA_DIR",
        Path(__file__).resolve().parent.parent / "data",
    )


def corpus_dir() -> Path:
    """분봉 corpus 폴더. FIBTRADER_CORPUS_DIR 또는 data_dir()/corpus."""
    return _resolve("FIBTRADER_CORPUS_DIR", data_dir() / "corpus")


def artifacts_dir() -> Path:
    """학습 산출물(모델, oof, notebook) 폴더. FIBTRADER_ARTIFACTS_DIR 또는 data_dir()/artifacts."""
    out = _resolve("FIBTRADER_ARTIFACTS_DIR", data_dir() / "artifacts")
    out.mkdir(parents=True, exist_ok=True)
    return out
