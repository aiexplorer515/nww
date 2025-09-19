import streamlit as st
import pandas as pd
import plotly.express as px
from nwwpkg.utils.loader import load_jsonl
from nwwpkg.utils.adapters import normalize_dataframe, safe_float

st.title("🟩 Event Blocks")

bundle_id = st.session_state.get("bundle_id", "sample")
path = f"data/bundles/{bundle_id}/event_blocks.jsonl"

records = load_jsonl(path)
if not records:
    st.warning("⚠️ 이벤트 블록 데이터 없음")
else:
    df = pd.DataFrame(records)
    df = normalize_dataframe(df, {
        "x": ["x", "dim1"],
        "y": ["y", "dim2"],
        "cluster": ["cluster", "group"],
        "risk": ["risk", "score"]
    })
    df["risk"] = df["risk"].apply(lambda x: safe_float(x, 0))
    fig = px.scatter(df, x="x", y="y", color="cluster", size="risk", title="Event Block Map")
    st.plotly_chart(fig, use_container_width=True)
