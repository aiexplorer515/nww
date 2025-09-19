import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe

st.title("⚙️ Normalize")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/normalized.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 정규화 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "id": ["id"],
        "text": ["text", "content", "body"],
        "lang": ["lang", "language"]
    })
    st.dataframe(df.head(20))
