import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl

st.title("📜 Ledger")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/ledger.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 원장 데이터 없음")
else:
    st.json(records[:20])
