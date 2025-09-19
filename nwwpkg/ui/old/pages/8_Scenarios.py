import streamlit as st
import pandas as pd
from nwwpkg.utils.adapters import safe_float, normalize_dataframe
from nwwpkg.utils.loader import load_jsonl

st.title("📄 Scenarios")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/scenarios.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 시나리오 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "title": ["title", "scenario", "name"],
        "score": ["score", "risk", "value"],
        "description": ["desc", "explanation", "summary"]
    })

    for _, r in df.iterrows():
        st.markdown(
            f"**{r.get('title','시나리오')}** - 위험 {safe_float(r.get('score'),0):.2f}"
        )
        if r.get("description"):
            st.caption(r["description"])
