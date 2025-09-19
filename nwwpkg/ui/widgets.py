# nwwpkg/ui/widgets.py 일부
import streamlit as st, pandas as pd, re

def evidence_view(df: pd.DataFrame, text_col="clean_text", keys=None, n=8):
    keys = keys or []
    def hi(t): 
        for k in keys: t = re.sub(f"({re.escape(k)})", r"**\1**", t, flags=re.I)
        return t
    sdf = df.sample(min(n, len(df))).copy()
    sdf[text_col] = sdf[text_col].fillna("").map(hi)
    st.dataframe(sdf[[text_col]].rename(columns={text_col:"본문(근거 하이라이트)"}), use_container_width=True)
