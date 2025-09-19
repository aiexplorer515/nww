import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl

st.title("🧬 Fusion (통합 결과)")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/fusion.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ Fusion 데이터 없음")
else:
    st.json(records[:10])
