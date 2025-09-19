import streamlit as st
import pandas as pd
from pathlib import Path
from nwwpkg.ui.components.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="EDS Stage", layout="wide")
st.title("📚 Expert Data System (EDS)")

render_sidebar_nav()

bundle_id = st.sidebar.text_input("Bundle ID", "sample")
eds_file = Path(f"data/bundles/{bundle_id}/eds.jsonl")

if st.button("▶️ Run EDS"):
    st.info("EDS 실행 (stub) → eds.jsonl 생성됨")

if eds_file.exists():
    df = pd.read_json(eds_file, lines=True)
    st.dataframe(df.head(20))
