# nwwpkg/ui/dashboard/risk_blocks.py
import streamlit as st
import pandas as pd
import plotly.express as px

from nwwpkg.utils.scoring import backfill_fused_score
from nwwpkg.utils.style import next_palette

def render_risk_blocks(df_blocks: pd.DataFrame, bundle_id: str):
    """
    Render Risk Blocks section: 블록별 평균 위험도 Top-N
    """
    st.markdown("### 🔥 Risk Blocks (위험 블록)")

    if df_blocks is None or df_blocks.empty:
        st.info("⚠️ Risk Blocks 데이터가 없습니다. (blocks.jsonl 필요)")
        return

    # fused_score 백필
    df_blocks = backfill_fused_score(df_blocks, bundle_id)

    if "fused_score" not in df_blocks.columns or "block" not in df_blocks.columns:
        st.warning("⚠️ blocks 데이터에 'fused_score' 또는 'block' 컬럼이 없습니다.")
        return

    # 블록별 평균 위험도 Top-N
    topn = st.slider("표시 개수(Top-N)", 5, 50, 10, 5, key="topn_blocks")
    top_blk = (
        df_blocks.groupby("block", observed=True)["fused_score"]
                 .mean().reset_index()
                 .sort_values("fused_score", ascending=False)
                 .head(topn)
    )

    tabs_blk = st.tabs(["📊 차트", "📄 표"])
    with tabs_blk[0]:
        fig_blk = px.bar(
            top_blk, x="block", y="fused_score", text="fused_score",
            title=f"블록별 평균 위험도 (Top {topn})",
            labels={"block": "블록", "fused_score": "평균 위험도"},
            color_discrete_sequence=next_palette()
        )
        fig_blk.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_blk, use_container_width=True)

    with tabs_blk[1]:
        st.dataframe(
            top_blk.rename(columns={"block": "블록", "fused_score": "평균 위험도"}),
            use_container_width=True
        )
