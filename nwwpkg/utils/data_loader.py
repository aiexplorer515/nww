from pathlib import Path
import pandas as pd
from typing import Optional, List

BASE = Path("data/bundles")

def _p(bundle_id: str, name: str) -> Path:
    return BASE / bundle_id / name

def load_jsonl(bundle_id: str, name: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """data/bundles/{bundle_id}/{name} 읽어서 DataFrame 반환. 없으면 빈 DF."""
    f = _p(bundle_id, name)
    if not f.exists():
        return pd.DataFrame(columns=cols or [])
    try:
        df = pd.read_json(f, lines=True)
        if cols:
            for c in cols:
                if c not in df.columns: df[c] = None
        return df
    except Exception:
        return pd.DataFrame(columns=cols or [])

def save_jsonl(df: pd.DataFrame, bundle_id: str, name: str) -> Path:
    out = _p(bundle_id, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", lines=True, force_ascii=False)
    return out
