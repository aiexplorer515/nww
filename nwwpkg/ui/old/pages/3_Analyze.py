import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe

st.title("🔎 Analyze")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/analysis.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 분석 결과 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "id": ["id"],
        "keywords": ["keywords", "keyphrases"],
        "entities": ["entities", "actors"]
    })
    st.json(df.head(10).to_dict(orient="records"))
