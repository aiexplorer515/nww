# nwwpkg/utils/io.py
import json
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict


def load_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    """JSONL 파일을 pandas DataFrame으로 로드"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def save_jsonl(path: Union[str, Path], rows: Union[List[Dict], pd.DataFrame]):
    """리스트(dict) 또는 DataFrame을 JSONL 파일로 저장"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # DataFrame이면 dict 리스트로 변환
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")

    if not isinstance(rows, list):
        raise TypeError(f"rows should be a list of dicts, got {type(rows)}")

    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Each row must be a dict, got {type(row)}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_lines(path: Union[str, Path]) -> int:
    """(내부용) 파일 라인 수 카운트"""
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_lines(path: Union[str, Path]) -> int:
    """(외부 공개용) 파일 라인 수 카운트"""
    return _count_lines(path)


def ensure_text_column(df: pd.DataFrame):
    """데이터프레임에서 본문 텍스트 컬럼 자동 탐지"""
    candidate_cols = ["text", "clean_text", "body", "normalized"]

    for col in candidate_cols:
        if col in df.columns:
            return col

    # fallback: 문자열 컬럼 중 가장 긴 평균 길이
    text_like_cols = df.select_dtypes(include=["object"]).columns
    if len(text_like_cols) > 0:
        avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_like_cols}
        if avg_lengths:
            return max(avg_lengths, key=avg_lengths.get)

    return None
