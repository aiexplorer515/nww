import streamlit as st
import pandas as pd
from pathlib import Path
from nwwpkg.ui.components.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Scoring Stage", layout="wide")
st.title("⚖️ Scoring 단계")

render_sidebar_nav()

bundle_id = st.sidebar.text_input("Bundle ID", "sample")
scores_file = Path(f"data/bundles/{bundle_id}/scores.jsonl")

if st.button("▶️ Run Scoring"):
    st.info("스코어링 실행 (stub) → scores.jsonl 생성됨")

if scores_file.exists():
    df = pd.read_json(scores_file, lines=True)
    st.dataframe(df.head(20))
    if "score" in df:
        st.line_chart(df["score"])
