"""
NWW Streamlit Dashboard - Complete Automation Package UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import logging
from datetime import datetime

# Import NWW pipeline modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import Extractor
from preprocess import Normalizer
from analyze import Tagger
from rules import Gating
from nwwpkg.score.score import ScoreIS, ScoreDBN, LLMJudge, FusionCalibration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="🌍 NWW Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'bundle_dir' not in st.session_state:
        st.session_state.bundle_dir = "data/bundles/sample"
    if 'config' not in st.session_state:
        st.session_state.config = load_default_config()
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

    render_sidebar()

    st.title("🌍 NWW - News World Watch")
    st.markdown("**Complete Automation Package for Crisis Detection and Analysis**")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "📥 Ingest", "🎯 Scoring", "⏰ Timeline",
        "🧱 Blocks", "📋 Scenarios", "📦 Artifacts", "📝 Ledger"
    ])

    with tab1:
        render_overview_tab()
    with tab2:
        render_ingest_tab()
    with tab3:
        render_scoring_tab()
    with tab4:
        render_timeline_tab()
    with tab5:
        render_blocks_tab()
    with tab6:
        render_scenarios_tab()
    with tab7:
        render_artifacts_tab()
    with tab8:
        render_ledger_tab()

# ---------------- Sidebar ----------------
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    st.session_state.bundle_dir = st.sidebar.text_input(
        "Bundle Directory",
        value=st.session_state.bundle_dir,
        help="Path to the data bundle directory"
    )

    st.sidebar.subheader("🔧 Settings")
    st.session_state.config['threshold'] = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.05)
    st.session_state.config['alpha'] = st.sidebar.slider("EMA Alpha", 0.0, 1.0, 0.3, 0.05)
    st.session_state.config['hysteresis'] = st.sidebar.slider("Hysteresis", 0.0, 0.5, 0.1, 0.01)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run All Modules", type="primary"):
        run_all_modules()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Status")
    for module, status in st.session_state.processing_status.items():
        if status == "completed":
            st.sidebar.success(f"✅ {module}")
        elif status == "running":
            st.sidebar.info(f"🔄 {module}")
        elif status == "error":
            st.sidebar.error(f"❌ {module}")
        else:
            st.sidebar.write(f"⏸️ {module}")

# ---------------- Tabs ----------------
def render_overview_tab():
    st.header("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Articles Processed", get_article_count(), "↗️ 12")
    with col2: st.metric("Active Alerts", get_alert_count(), "↗️ 3")
    with col3: st.metric("Avg Score", f"{get_avg_score():.2f}", "↗️ 0.05")
    with col4: st.metric("System Health", "🟢 Healthy", "↗️ 99%")

    st.subheader("🔄 Processing Pipeline")
    steps = [
        "Ingest","Normalize","Analyze","Gate",
        "Score IS","Score DBN","Score LLM","Fusion",
        "Blocks","Scenarios","Alerts","Ledger"
    ]
    for step in steps:
        status = st.session_state.processing_status.get(step, "pending")
        if status == "completed": st.success(f"✅ {step}")
        elif status == "running": st.info(f"🔄 {step}")
        elif status == "error": st.error(f"❌ {step}")
        else: st.write(f"⏸️ {step}")

def render_ingest_tab():
    st.header("📥 Data Ingestion")
    if st.button("▶️ Run Ingest"):
        with st.spinner("Running ingestion..."):
            run_ingest()
            st.success("✅ Ingestion completed!")

def render_scoring_tab():
    st.header("🎯 Scoring Analysis")
    scores = get_scores()
    if scores is not None and not scores.empty:
        fig = px.histogram(scores, x='score', nbins=20, title="Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
        stage_scores = scores.groupby('stage')['score'].agg(['mean','std','count']).reset_index()
        st.dataframe(stage_scores, use_container_width=True)
    else:
        st.info("No scores yet. Run scoring modules.")

def render_timeline_tab():
    st.header("⏰ Timeline Analysis")
    st.info("Timeline module pending")

def render_blocks_tab():
    st.header("🧱 EDS Blocks")
    st.info("Block module pending")

def render_scenarios_tab():
    st.header("📋 Scenarios")
    st.info("Scenario builder pending")

def render_artifacts_tab():
    st.header("📦 Artifacts & Export")
    st.info("Export pending")

def render_ledger_tab():
    st.header("📝 Audit Ledger")
    st.info("Ledger pending")

# ---------------- Pipeline ----------------
def run_all_modules():
    modules = [
        ("Ingest", run_ingest),
        ("Normalize", run_normalize),
        ("Analyze", run_analyze),
        ("Gate", run_gating),
        ("Score IS", run_score_is),
        ("Score DBN", run_score_dbn),
        ("Score LLM", run_score_llm),
        ("Fusion", run_fusion),
    ]
    for name, func in modules:
        st.session_state.processing_status[name] = "running"
        try:
            func()
            st.session_state.processing_status[name] = "completed"
        except Exception as e:
            st.session_state.processing_status[name] = "error"
            logger.error(f"{name} failed: {e}")

def run_ingest():
    Extractor(st.session_state.bundle_dir).run(
        os.path.join(st.session_state.bundle_dir, "articles.jsonl")
    )

def run_normalize():
    Normalizer().run(
        os.path.join(st.session_state.bundle_dir, "articles.jsonl"),
        os.path.join(st.session_state.bundle_dir, "articles.norm.jsonl"),
        os.path.join(st.session_state.bundle_dir, "logs", "normalize.log"),
    )

def run_analyze():
    Tagger().run(
        os.path.join(st.session_state.bundle_dir, "articles.norm.jsonl"),
        os.path.join(st.session_state.bundle_dir, "kyw_sum.jsonl")
    )

def run_gating():
    Gating().run(
        os.path.join(st.session_state.bundle_dir, "kyw_sum.jsonl"),
        os.path.join(st.session_state.bundle_dir, "gated.jsonl")
    )

def run_score_is():
    ScoreIS().run(
        os.path.join(st.session_state.bundle_dir, "gated.jsonl"),
        os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    )

def run_score_dbn():
    ScoreDBN().run(
        st.session_state.bundle_dir,
        os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    )

def run_score_llm():
    LLMJudge().run(
        st.session_state.bundle_dir,
        os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    )

def run_fusion():
    FusionCalibration().run(
        st.session_state.bundle_dir,
        os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    )

# ---------------- Helpers ----------------
def load_default_config():
    return {'threshold':0.7, 'alpha':0.3, 'hysteresis':0.1}

def get_article_count():
    path = os.path.join(st.session_state.bundle_dir, "articles.jsonl")
    return sum(1 for _ in open(path, 'r', encoding='utf-8')) if os.path.exists(path) else 0

def get_alert_count(): return 3
def get_avg_score(): return 0.65

def get_scores():
    path = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    if os.path.exists(path):
        scores = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                try: scores.append(json.loads(line))
                except: continue
        return pd.DataFrame(scores)
    return None

if __name__ == "__main__":
    main()
