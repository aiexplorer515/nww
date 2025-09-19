import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe

st.title("🧱 Expert Blocks (EDS)")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/blocks.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 블록 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "block_id": ["id", "block_id"],
        "name": ["name", "title"],
        "domain": ["domain", "category"]
    })
    st.dataframe(df.head(20))
