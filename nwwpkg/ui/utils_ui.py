import pandas as pd
from contextlib import contextmanager
import time, streamlit as st

def safe_df(obj) -> pd.DataFrame:
    if obj is None: return pd.DataFrame()
    if isinstance(obj, pd.DataFrame): return obj
    return pd.DataFrame(obj)

def safe_series(df: pd.DataFrame, col: str, dtype="float64") -> pd.Series:
    df = safe_df(df)
    if df.empty:
        return pd.Series(dtype=dtype)
    import pandas as pd
    s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(0, index=df.index, dtype=dtype)
    return s.fillna(0)

@contextmanager
def section(title: str):
    st.subheader(title)
    start = time.time()
    try:
        yield
    except Exception as e:
        st.error(f"⚠️ 섹션 오류: {type(e).__name__}: {str(e)[:180]}…")
    finally:
        dur = int((time.time()-start)*1000)
        st.caption(f"⏱ {dur} ms")
