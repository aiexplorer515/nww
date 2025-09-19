import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import os

from nwwpkg.utils.pipeline import run_full_pipeline, run_pipeline

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(page_title="NWW Dashboard", layout="wide")

# ---------------------------
# Session state for routing
# ---------------------------
if "stage" not in st.session_state:
    st.session_state["stage"] = "overview"

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ 설정")
alert_level = st.sidebar.slider("경보 기준 점수", 0.0, 1.0, 0.7, 0.05)
bundle_id = st.sidebar.text_input("분석 Bundle ID", "sample")

if st.sidebar.button("▶️ Run Full Pipeline"):
    out_file = run_full_pipeline(bundle_id)
    st.sidebar.success(f"✅ 전체 파이프라인 완료 → {out_file}")

# 사이드바 네비게이션 버튼
st.sidebar.markdown("---")
st.sidebar.subheader("📌 단계 이동")

stage_map = {
    "overview": "📊 Dashboard",
    "ingest": "📥 Ingest",
    "preprocess": "🧹 Preprocess",
    "analysis": "🔍 Analysis",
    "scoring": "⚖️ Scoring",
    "scenarios": "📑 Scenarios",
    "alerts": "🚨 Alerts",
    "eds": "📚 EDS"
}

for key, label in stage_map.items():
    if st.sidebar.button(label, key=f"nav_{key}"):
        st.session_state["stage"] = key

stage = st.session_state["stage"]

# ---------------------------
# Stage Rendering
# ---------------------------

# ====== 1. OVERVIEW ======
if stage == "overview":
    st.title("🌐 Crisis Overview")

    bundle_dir = Path(f"data/bundles/{bundle_id}")
    latest_file = None
    if bundle_dir.exists():
        files = sorted(bundle_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if files:
            latest_file = files[0]

    df = pd.read_json(latest_file, lines=True) if latest_file and latest_file.exists() else pd.DataFrame()

    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    total_articles = len(df)
    active_alerts = (df["score"] > alert_level).sum() if "score" in df else 0
    avg_score = df["score"].mean() if "score" in df else 0.0
    system_status = "✅ 정상" if total_articles > 0 else "❌ 데이터 없음"

    col1.metric("기사 누적 수", f"{total_articles:,}")
    col2.metric("활성 경보 수", active_alerts)
    col3.metric("평균 점수", f"{avg_score:.2f}")
    col4.metric("시스템 상태", system_status)

    st.markdown("---")

    # 지역별/도메인별 현황
    if not df.empty:
        col1, col2 = st.columns(2)
        if "region" in df:
            col1.plotly_chart(px.bar(df, x="region", title="지역별 기사 수"), use_container_width=True)
        if "domain" in df:
            col2.plotly_chart(px.bar(df, x="domain", title="도메인별 기사 수"), use_container_width=True)

        st.markdown("---")

        # 프레임 통계
        col1, col2 = st.columns(2)
        if "frame" in df:
            col1.plotly_chart(px.pie(df, names="frame", title="프레임 분포"), use_container_width=True)
        if "date" in df and "frame" in df:
            col2.plotly_chart(
                px.area(df, x="date", y="score", color="frame", title="프레임 시계열 추이"),
                use_container_width=True
            )

        st.markdown("### 🔝 Top 기사 리스트")
        cols = [c for c in ["title", "frame", "score"] if c in df]
        if cols:
            st.table(df[cols].head(20))

        st.markdown("---")

        # 네트워크 그래프
        st.subheader("🌐 관계 네트워크 그래프")
        if "frame" in df and "region" in df:
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_node(row["frame"], type="frame")
                G.add_node(row["region"], type="region")
                G.add_edge(row["frame"], row["region"])

            pos = nx.spring_layout(G, k=0.5)
            edge_x, edge_y, node_x, node_y, node_text = [], [], [], [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="gray")))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                     marker=dict(size=10, color="skyblue"),
                                     text=node_text, textposition="top center"))
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 위험도 히트맵
        st.subheader("🔥 위험도 히트맵")
        if "region" in df and "frame" in df and "score" in df:
            pivot = df.pivot_table(index="region", columns="frame", values="score", aggfunc="mean").fillna(0)
            fig = px.imshow(
                pivot,
                labels=dict(x="프레임", y="지역", color="평균 점수"),
                aspect="auto",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)

# ====== 2. INGEST ======
elif stage == "ingest":
    st.title("📥 Ingest 단계")
    uploaded = st.file_uploader("기사 JSONL 업로드", type=["jsonl"])
    if uploaded:
        df = pd.read_json(uploaded, lines=True)
        st.dataframe(df.head(20))
        st.success("✅ 데이터 로드 완료")
    if st.button("▶️ Run Ingest"):
        out_file = run_pipeline("ingest", f"data/bundles/{bundle_id}")
        st.success(f"Ingest 완료 → {out_file}")

# ====== 3. PREPROCESS ======
elif stage == "preprocess":
    st.title("🧹 Preprocess 단계")
    if st.button("▶️ Run Preprocess"):
        out_file = run_pipeline("preprocess", f"data/bundles/{bundle_id}")
        st.success(f"Preprocess 완료 → {out_file}")
    norm_file = Path(f"data/bundles/{bundle_id}/articles.norm.jsonl")
    if norm_file.exists():
        df = pd.read_json(norm_file, lines=True)
        st.dataframe(df.head(20))

# ====== 4. ANALYSIS ======
elif stage == "analysis":
    st.title("🔍 Analysis 단계")
    if st.button("▶️ Run Analysis"):
        out_file = run_pipeline("analysis", f"data/bundles/{bundle_id}")
        st.success(f"Analysis 완료 → {out_file}")
    kyw_file = Path(f"data/bundles/{bundle_id}/kyw_sum.jsonl")
    if kyw_file.exists():
        df = pd.read_json(kyw_file, lines=True)
        st.dataframe(df.head(20))
        if "frame" in df:
            st.bar_chart(df["frame"].value_counts())

# ====== 5. SCORING ======
elif stage == "scoring":
    st.title("⚖️ Scoring 단계")
    if st.button("▶️ Run Scoring"):
        out_file = run_pipeline("scoring", f"data/bundles/{bundle_id}")
        st.success(f"Scoring 완료 → {out_file}")
    scores_file = Path(f"data/bundles/{bundle_id}/scores.jsonl")
    if scores_file.exists():
        df = pd.read_json(scores_file, lines=True)
        st.dataframe(df.head(20))
        if "score" in df:
            st.line_chart(df["score"])

# ====== 6. SCENARIOS ======
elif stage == "scenarios":
    st.title("📑 Scenarios 단계")
    if st.button("▶️ Run Scenario Generation"):
        out_file = run_pipeline("scenarios", f"data/bundles/{bundle_id}")
        st.success(f"Scenarios 완료 → {out_file}")
    scenario_file = Path(f"data/bundles/{bundle_id}/scenarios.jsonl")
    if scenario_file.exists():
        df = pd.read_json(scenario_file, lines=True)
        st.dataframe(df.head(20))

# ====== 7. ALERTS ======
elif stage == "alerts":
    st.title("🚨 Alerts 단계")
    alerts_file = Path(f"data/bundles/{bundle_id}/alerts.jsonl")
    if alerts_file.exists():
        df = pd.read_json(alerts_file, lines=True)
        st.table(df.head(20))
    st.info("경보 기준: Score > Alert Level")

# ====== 8. EDS ======
elif stage == "eds":
    st.title("📚 Expert Data System (EDS)")
    eds_file = Path(f"data/bundles/{bundle_id}/eds.jsonl")
    if eds_file.exists():
        df = pd.read_json(eds_file, lines=True)
        st.dataframe(df.head(20))
    st.info("EDS 모드: 전문가 피드백/체크리스트 검토 기능")
