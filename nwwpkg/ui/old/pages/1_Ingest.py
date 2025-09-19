import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe

st.title("📰 Ingest")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/articles.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "id": ["id", "article_id"],
        "title": ["title", "headline"],
        "url": ["url", "link"]
    })
    st.dataframe(df[["id", "title", "url"]].head(20))
