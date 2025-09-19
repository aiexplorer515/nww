# -*- coding: utf-8 -*-
"""
Ledger(viewer): JSONL 로드
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

def load(bundle_id: str, filename: str) -> pd.DataFrame:
    p = Path(f"data/{bundle_id}/{filename}")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_json(p, orient="records", lines=True)
