"""
NWW Streamlit Dashboard - Crisis Detection with Landing Page + URL Input
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import logging
from datetime import datetime, timedelta

# 모듈 불러오기 경로 설정
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import Extractor
from preprocess import Normalizer
from analyze import Tagger
from rules import Gating
from score import ScoreIS, ScoreDBN, LLMJudge

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------- MAIN -------------------------
def main():
    st.set_page_config(
        page_title="🌍 NWW Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 세션 상태 초기화
    if "bundle_dir" not in st.session_state:
        st.session_state.bundle_dir = "data/bundles/sample"
    if "config" not in st.session_state:
        st.session_state.config = load_default_config()
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {}

    # 사이드바
    render_sidebar()

    # 랜딩페이지
    st.title("🌍 NWW Crisis Detection Dashboard")
    st.markdown("**실시간 위기 신호를 지역별·분야별로 종합 표시합니다.**")
    render_landing_dashboard()

    # 탭
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "📥 Ingest", "🎯 Scoring", "⏰ Timeline",
        "🧱 Blocks", "📋 Scenarios", "📦 Artifacts", "📝 Ledger"
    ])
    with tab1: render_overview_tab()
    with tab2: render_ingest_tab()
    with tab3: render_scoring_tab()
    with tab4: render_timeline_tab()
    with tab5: render_blocks_tab()
    with tab6: render_scenarios_tab()
    with tab7: render_artifacts_tab()
    with tab8: render_ledger_tab()


# ------------------------- 랜딩 페이지 -------------------------
def render_landing_dashboard():
    """위기 종합 대시보드"""
    st.header("📡 Crisis Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 지역별 위기 현황")
        region_data = pd.DataFrame({
            "Region": ["Asia", "Europe", "Middle East", "Africa", "Americas"],
            "Active Alerts": [5, 3, 7, 2, 4]
        })
        fig = px.bar(region_data, x="Region", y="Active Alerts", color="Region")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("⚔️ 분야별 위기 현황")
        domain_data = pd.DataFrame({
            "Domain": ["Military", "Diplomacy", "Economy"],
            "Risk Level": [0.82, 0.65, 0.73]
        })
        fig2 = px.pie(domain_data, names="Domain", values="Risk Level",
                      title="위험도 분포")
        st.plotly_chart(fig2, width="stretch")

    st.markdown("🔔 **현재 활성 알림:** 15건 / 평균 위험도 0.74")


# ------------------------- 사이드바 -------------------------
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    bundle_dir = st.sidebar.text_input("Bundle Directory",
                                       value=st.session_state.bundle_dir,
                                       help="데이터 번들 디렉토리 경로")
    st.session_state.bundle_dir = bundle_dir

    st.sidebar.subheader("🔧 Settings")
    st.session_state.config["threshold"] = st.sidebar.slider(
        "Alert Threshold", 0.0, 1.0, 0.7, 0.05)
    st.session_state.config["alpha"] = st.sidebar.slider(
        "EMA Alpha", 0.0, 1.0, 0.3, 0.05)
    st.session_state.config["hysteresis"] = st.sidebar.slider(
        "Hysteresis", 0.0, 0.5, 0.1, 0.01)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run All Modules", type="primary"):
        run_all_modules()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Status")
    for module, status in st.session_state.processing_status.items():
        if status == "completed": st.sidebar.success(f"✅ {module}")
        elif status == "running": st.sidebar.info(f"🔄 {module}")
        elif status == "error": st.sidebar.error(f"❌ {module}")
        else: st.sidebar.write(f"⏸️ {module}")


# ------------------------- INGEST 탭 -------------------------
def render_ingest_tab():
    st.header("📥 Data Ingestion")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 Sources")
        sources = get_sources()
        if sources is not None and not sources.empty:
            st.dataframe(sources, width="stretch")
        else:
            st.info("No sources found. Please add sources.")

        # URL 입력 form
        with st.form("add_url_form"):
            new_url = st.text_input("뉴스 기사 URL 입력", "")
            submitted = st.form_submit_button("➕ Add URL")
            if submitted and new_url:
                save_url(new_url)
                st.success(f"✅ URL 추가됨: {new_url}")

    with col2:
        st.subheader("⚙️ Controls")
        if st.button("▶️ Run Ingest"):
            with st.spinner("Running ingestion..."):
                try:
                    run_ingest()
                    st.success("✅ Ingestion completed!")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")


def save_url(url: str):
    """입력된 URL을 sources.csv에 저장"""
    sources_path = os.path.join(st.session_state.bundle_dir, "sources.csv")
    os.makedirs(os.path.dirname(sources_path), exist_ok=True)

    if os.path.exists(sources_path):
        df = pd.read_csv(sources_path)
    else:
        df = pd.DataFrame(columns=["URL"])

    if url not in df["URL"].values:
        df.loc[len(df)] = [url]
        df.to_csv(sources_path, index=False, encoding="utf-8")


# ------------------------- PLACEHOLDER (다른 탭 유지) -------------------------
def render_overview_tab(): st.header("📊 Overview (기존 내용 유지)")
def render_scoring_tab(): st.header("🎯 Scoring (기존 내용 유지)")
def render_timeline_tab(): st.header("⏰ Timeline (기존 내용 유지)")
def render_blocks_tab(): st.header("🧱 Blocks (기존 내용 유지)")
def render_scenarios_tab(): st.header("📋 Scenarios (기존 내용 유지)")
def render_artifacts_tab(): st.header("📦 Artifacts (기존 내용 유지)")
def render_ledger_tab(): st.header("📝 Ledger (기존 내용 유지)")


# ------------------------- MODULE 실행 함수 -------------------------
def run_all_modules():
    modules = [
        ("Ingest", run_ingest),
        ("Normalize", run_normalize),
        ("Analyze", run_analyze),
        ("Gate", run_gating),
        ("Score IS", run_score_is),
        ("Score DBN", run_score_dbn),
        ("Score LLM", run_score_llm),
    ]
    progress = st.progress(0)
    for i, (name, func) in enumerate(modules):
        st.session_state.processing_status[name] = "running"
        try:
            func()
            st.session_state.processing_status[name] = "completed"
        except Exception as e:
            st.session_state.processing_status[name] = "error"
            logger.error(f"Error in {name}: {e}")
        progress.progress((i + 1) / len(modules))


def run_ingest():
    extractor = Extractor(st.session_state.bundle_dir)
    output_path = os.path.join(st.session_state.bundle_dir, "articles.jsonl")
    extractor.run(output_path)


def run_normalize():
    normalizer = Normalizer()
    inp = os.path.join(st.session_state.bundle_dir, "articles.jsonl")
    out = os.path.join(st.session_state.bundle_dir, "articles.norm.jsonl")
    normalizer.run(inp, out, None)


def run_analyze():
    tagger = Tagger()
    inp = os.path.join(st.session_state.bundle_dir, "articles.norm.jsonl")
    out = os.path.join(st.session_state.bundle_dir, "kyw_sum.jsonl")
    tagger.run(inp, out)


def run_gating():
    gating = Gating()
    inp = os.path.join(st.session_state.bundle_dir, "kyw_sum.jsonl")
    out = os.path.join(st.session_state.bundle_dir, "gated.jsonl")
    gating.run(inp, out)


def run_score_is():
    scorer = ScoreIS()
    inp = os.path.join(st.session_state.bundle_dir, "gated.jsonl")
    out = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    scorer.run(inp, out)


def run_score_dbn():
    scorer = ScoreDBN()
    out = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    scorer.run(st.session_state.bundle_dir, out)


def run_score_llm():
    scorer = LLMJudge()
    out = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    scorer.run(st.session_state.bundle_dir, out)


# ------------------------- UTILS -------------------------
def load_default_config():
    return {"threshold": 0.7, "alpha": 0.3, "hysteresis": 0.1}


def get_sources():
    try:
        sources_path = os.path.join(st.session_state.bundle_dir, "sources.csv")
        if os.path.exists(sources_path):
            return pd.read_csv(sources_path)
    except: return None
    return None


if __name__ == "__main__":
    main()
