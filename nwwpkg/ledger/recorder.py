# -*- coding: utf-8 -*-
"""
Ledger(recorder): JSONL append + dedup(선택) 저장
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save(bundle_id: str, filename: str, df: pd.DataFrame, dedup_on: str | None = None) -> str:
    out_dir = Path(f"data/{bundle_id}")
    _ensure_dir(out_dir)
    out = out_dir / filename

    if out.exists():
        old = pd.read_json(out, orient="records", lines=True)
        df = pd.concat([old, df], ignore_index=True)

    if dedup_on and dedup_on in df.columns:
        df = df.drop_duplicates(subset=[dedup_on], keep="last")

    df.to_json(out, orient="records", lines=True, force_ascii=False)
    return str(out)
