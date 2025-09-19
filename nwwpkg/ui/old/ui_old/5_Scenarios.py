import streamlit as st
import pandas as pd
from pathlib import Path
from nwwpkg.ui.components.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Scenarios Stage", layout="wide")
st.title("📑 Scenarios 단계")

render_sidebar_nav()

bundle_id = st.sidebar.text_input("Bundle ID", "sample")
scenario_file = Path(f"data/bundles/{bundle_id}/scenarios.jsonl")

if st.button("▶️ Run Scenario Generation"):
    st.info("시나리오 생성 실행 (stub) → scenarios.jsonl 생성됨")

if scenario_file.exists():
    df = pd.read_json(scenario_file, lines=True)
    st.dataframe(df.head(20))
