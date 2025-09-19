# nwwpkg/ui/dashboard/sources.py
import streamlit as st
import pandas as pd
import plotly.express as px
from nwwpkg.utils.style import next_palette


def render_sources(df_ingest: pd.DataFrame):
    """
    수집된 기사들의 언론사/출처(Source) 분포 시각화
    - df_ingest: ingest.jsonl DataFrame (source 컬럼 필요)
    """
    st.subheader("📰 Sources (언론사/출처 분포)")

    if df_ingest is None or df_ingest.empty or "source" not in df_ingest.columns:
        st.info("Ingest 데이터에 'source' 컬럼이 없습니다.")
        return

    # 출처별 건수 집계
    src = df_ingest["source"].fillna("Manual").value_counts().reset_index()
    src.columns = ["source", "count"]

    src_tabs = st.tabs(["📊 차트", "📄 표"])

    # --- 차트 ---
    with src_tabs[0]:
        fig_src = px.bar(
            src.head(30),
            x="source",
            y="count",
            text="count",
            title="언론사/도메인 분포 (Top 30)",
            labels={"source": "소스", "count": "건수"},
            color_discrete_sequence=next_palette()
        )
        fig_src.update_traces(textposition="outside")
        st.plotly_chart(fig_src, use_container_width=True)

    # --- 표 ---
    with src_tabs[1]:
        st.dataframe(src, use_container_width=True)
