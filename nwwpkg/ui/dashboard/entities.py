# nwwpkg/ui/dashboard/entities.py
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from nwwpkg.utils.style import next_palette


def render_entities(df_ana: pd.DataFrame):
    """
    주요 인물/조직(Entities) Top-N 시각화
    - df_ana: analyze.jsonl DataFrame (network 컬럼 포함 필요)
    """
    st.subheader("🏷️ Entities (주요 인물/조직)")

    ent_tabs = st.tabs(["📊 차트", "📄 표"])

    if df_ana is None or df_ana.empty or "network" not in df_ana.columns:
        with ent_tabs[0]:
            st.info("Analyze 데이터에 'network' 컬럼이 없어 엔티티 Top-N을 생성할 수 없습니다.")
        with ent_tabs[1]:
            st.empty()
        return

    counter = Counter()
    for g in df_ana["network"].dropna():
        try:
            nodes = (g or {}).get("nodes", {})
            counter.update({k: int(v) for k, v in nodes.items()})
        except Exception:
            continue

    if not counter:
        st.info("네트워크 데이터에서 유효한 엔티티를 찾지 못했습니다.")
        return

    # Top-N 개수 조절
    topn = st.slider("표시 개수", 5, 50, 20, 5, key="topn_entities")
    ent_df = pd.DataFrame(counter.most_common(topn), columns=["entity", "count"])

    # --- 차트 ---
    with ent_tabs[0]:
        fig_ent = px.bar(
            ent_df,
            x="entity",
            y="count",
            text="count",
            title=f"엔티티 상위 Top-{topn}",
            color_discrete_sequence=next_palette()
        )
        fig_ent.update_traces(textposition="outside")
        st.plotly_chart(fig_ent, use_container_width=True)

    # --- 표 ---
    with ent_tabs[1]:
        st.dataframe(ent_df, use_container_width=True)
