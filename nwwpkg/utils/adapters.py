# nwwpkg/utils/adapters.py
import pandas as pd

def safe_float(val, default=0.0):
    """NoneType, 문자열을 안전하게 float 변환"""
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    """NoneType, 문자열을 안전하게 int 변환"""
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def normalize_dataframe(df: pd.DataFrame, required_cols: dict):
    """
    DataFrame 컬럼을 표준 스키마로 정리
    required_cols: {표준컬럼명: [후보컬럼들]}
    """
    for target, candidates in required_cols.items():
        for cand in candidates:
            if cand in df.columns:
                df[target] = df[cand]
                break
        if target not in df.columns:
            df[target] = None
    return df
