# nwwpkg/ui/dashboard/scenarios.py
import streamlit as st
import pandas as pd
import plotly.express as px
from nwwpkg.utils.scoring import ensure_fused


def render_scenarios(df_scen: pd.DataFrame, bundle_id: str = None):
    """
    위험 시나리오 섹션 렌더링
    - df_scen: scenarios.jsonl DataFrame
    - bundle_id: (선택) 백필 등 활용 시 필요
    """
    st.subheader("🧭 Scenarios (위험 시나리오)")

    if df_scen is None or df_scen.empty:
        st.info("Scenarios 데이터가 없습니다. (⚡ 백로그 일괄 처리 후 재시도 가능)")
        return

    # 위험도 보정
    df_scen = ensure_fused(df_scen, bundle_id)

    # 시나리오 이름 추출 유틸
    def _scen_name(v):
        if isinstance(v, dict):
            return v.get("scenario") or v.get("title") or str(v)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict):
                return first.get("title") or first.get("scenario") or str(first)
            return str(first)
        if isinstance(v, str):
            return v
        return "Unknown"

    # 시나리오 컬럼 결정
    scen_col = "scenario_predicted" if "scenario_predicted" in df_scen.columns else (
        "scenario_matched" if "scenario_matched" in df_scen.columns else None
    )

    if not scen_col:
        st.info("시나리오 컬럼(scenario_predicted / scenario_matched)이 없습니다.")
        return

    # 시나리오 이름 컬럼 생성
    tmp = df_scen.copy()
    tmp["scenario_name"] = tmp[scen_col].apply(_scen_name)

    # 평균 위험도 기준 Top-N
    topn = st.slider("표시 개수 (Top-N)", 5, 50, 20, 5, key="topn_scen")
    top_scen = (
        tmp.groupby("scenario_name", observed=True)["fused_score"]
        .mean().reset_index()
        .sort_values("fused_score", ascending=False)
        .head(topn)
    )

    tabs = st.tabs(["📊 차트", "📄 표", "📰 원자료 미리보기"])

    with tabs[0]:
        fig = px.bar(
            top_scen,
            x="scenario_name",
            y="fused_score",
            text="fused_score",
            title="시나리오별 평균 위험도 (Top-N)",
            labels={"scenario_name": "시나리오", "fused_score": "평균 위험도"},
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.dataframe(
            top_scen.rename(columns={"scenario_name": "시나리오", "fused_score": "평균 위험도"}),
            use_container_width=True,
        )

    with tabs[2]:
        cols = [c for c in ["date", "title", "scenario_predicted", "scenario_matched", "fused_score"] if c in tmp.columns]
        st.dataframe(tmp.sort_values("date", ascending=False)[cols].head(topn), use_container_width=True)
