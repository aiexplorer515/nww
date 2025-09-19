import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe, safe_float

st.title("📊 Scoring")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/scores.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 스코어 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "id": ["id"],
        "score": ["score", "risk"]
    })
    df["score"] = df["score"].apply(lambda x: safe_float(x, 0))
    st.bar_chart(df.set_index("id")["score"].head(20))
