# nwwpkg/utils/scoring.py
import pandas as pd

def backfill_fused_score(df: pd.DataFrame, bundle_id: str | None = None) -> pd.DataFrame:
    """fused_score 보강"""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "fused_score" in df.columns:
        return df

    candidates = ["score", "risk_score", "risk", "level_score", "alert_score"]
    for col in candidates:
        if col in df.columns:
            df["fused_score"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if bundle_id:
                print(f"[scoring] {bundle_id}: '{col}' → fused_score 보강 완료")
            return df

    df["fused_score"] = 0.0
    if bundle_id:
        print(f"[scoring] {bundle_id}: fused_score 보강 실패 → 기본값 0.0 적용")
    return df


def _ensure_fused(df: pd.DataFrame, bundle_id: str | None = None) -> pd.DataFrame:
    """항상 fused_score가 존재하도록 보장 (내부용)"""
    if df is None or df.empty:
        return df
    if "fused_score" not in df.columns:
        df = backfill_fused_score(df, bundle_id=bundle_id)
    return df

# ✅ 외부에서 쓰기 좋은 public alias
def ensure_fused(df: pd.DataFrame, bundle_id: str | None = None) -> pd.DataFrame:
    """항상 fused_score가 존재하도록 보장 (외부용)"""
    return _ensure_fused(df, bundle_id=bundle_id)


__all__ = ["backfill_fused_score", "_ensure_fused", "ensure_fused"]
