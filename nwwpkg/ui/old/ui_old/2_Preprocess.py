import streamlit as st
import pandas as pd
from pathlib import Path
from nwwpkg.ui.components.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Preprocess Stage", layout="wide")
st.title("🧹 Preprocess 단계")

render_sidebar_nav()

bundle_id = st.sidebar.text_input("Bundle ID", "sample")
norm_file = Path(f"data/bundles/{bundle_id}/articles.norm.jsonl")

if st.button("▶️ Run Preprocess"):
    # 실제 전처리 로직 연결
    st.info("전처리 실행 (stub) → norm 파일 생성됨")

if norm_file.exists():
    df = pd.read_json(norm_file, lines=True)
    st.dataframe(df.head(20))
