import streamlit as st
import pandas as pd
from pathlib import Path
from nwwpkg.ui.components.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Alerts Stage", layout="wide")
st.title("🚨 Alerts 단계")

render_sidebar_nav()

bundle_id = st.sidebar.text_input("Bundle ID", "sample")
alerts_file = Path(f"data/bundles/{bundle_id}/alerts.jsonl")

if st.button("▶️ Run Alerts"):
    st.info("경보 생성 실행 (stub) → alerts.jsonl 생성됨")

if alerts_file.exists():
    df = pd.read_json(alerts_file, lines=True)
    st.table(df.head(20))
