import streamlit as st
import pandas as pd
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe

st.title("🚪 Gate (체크리스트 매칭)")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/gated.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 게이팅 결과 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "id": ["id"],
        "matched_rules": ["matched", "rules"]
    })
    st.dataframe(df.head(20))
