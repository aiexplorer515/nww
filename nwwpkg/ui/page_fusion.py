# nwwpkg/ui/page_fusion.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def page_fusion(root: Path):
    st.header("⚖️ Fusion – 위험 점수 합성")

    score_file = root / "data/scores.jsonl"
    if not score_file.exists():
        st.warning("점수 데이터(scores.jsonl)가 없습니다.")
        return

    df = pd.read_json(score_file, lines=True)

    # 탭 구성
    tab1, tab2 = st.tabs(["📋 Table", "📊 Chart"])

    with tab1:
        st.subheader("위험 점수 테이블")
        st.dataframe(df)

    with tab2:
        st.subheader("위험 점수 시각화")
        fig = px.histogram(df, x="fused_score", nbins=20, title="위험 점수 분포")
        st.plotly_chart(fig, use_container_width=True)
