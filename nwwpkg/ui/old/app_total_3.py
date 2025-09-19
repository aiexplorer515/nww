"""
NWW Streamlit Dashboard - Complete Automation Package UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os, logging
from datetime import datetime

# Import NWW modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules that exist
try:
    from ingest import Extractor
    from preprocess import Normalizer
    from analyze import Tagger
    from rules import Gating
    from score import ScoreIS, ScoreDBN, LLMJudge
    from fusion import FusionCalibration
    from eds import EDSBlockMatcher
    from scenario import ScenarioBuilder
    from decider import AlertDecider
    from ledger import AuditLedger
    from eventblock import EventBlockAggregator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# -------------------------------------------------------------------
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Pipeline Runner
def run_pipeline(urls, out_dir="data/bundles/sample"):
    """Run the complete NWW pipeline."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    try:
    # 1) Ingest
        st.info("🔄 Step 1: Data Ingestion...")
        extractor = Extractor(out_dir)
        articles_path = os.path.join(out_dir, "articles.jsonl")
        extractor.run(articles_path)
        
        # Load articles for demo
        articles = []
        if os.path.exists(articles_path):
            with open(articles_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        articles.append(json.loads(line.strip()))
                    except:
                        continue

    # 2) Normalize
        st.info("🔄 Step 2: Text Normalization...")
        normalizer = Normalizer()
        norm_path = os.path.join(out_dir, "articles.norm.jsonl")
        log_path = os.path.join(out_dir, "logs", "normalize.log")
        normalizer.run(articles_path, norm_path, log_path)

        # 3) Analyze
        st.info("🔄 Step 3: Text Analysis...")
        tagger = Tagger()
        analysis_path = os.path.join(out_dir, "kyw_sum.jsonl")
        tagger.run(norm_path, analysis_path)

    # 4) Gating
        st.info("🔄 Step 4: Content Gating...")
        gating = Gating()
        gated_path = os.path.join(out_dir, "gated.jsonl")
        gating.run(analysis_path, gated_path)

    # 5) Scoring
        st.info("🔄 Step 5: Multi-modal Scoring...")
        scores_path = os.path.join(out_dir, "scores.jsonl")
        
        # IS Scoring
        score_is = ScoreIS()
        score_is.run(gated_path, scores_path)
        
        # DBN Scoring
        score_dbn = ScoreDBN()
        score_dbn.run(out_dir, scores_path)
        
        # LLM Scoring
        score_llm = LLMJudge()
        score_llm.run(out_dir, scores_path)

        # 6) Fusion
        st.info("🔄 Step 6: Score Fusion...")
        fusion = FusionCalibration()
        fused_path = os.path.join(out_dir, "fused_scores.jsonl")
        fusion.run(scores_path, fused_path)

        # 7) Blocks
        st.info("🔄 Step 7: EDS Block Matching...")
        block_matcher = EDSBlockMatcher()
        blocks_path = os.path.join(out_dir, "blocks.jsonl")
        block_matcher.run(out_dir, blocks_path)

        # 8) Scenarios
        st.info("🔄 Step 8: Scenario Construction...")
        scenario_builder = ScenarioBuilder()
        scenarios_path = os.path.join(out_dir, "scenarios.jsonl")
        scenario_builder.run(out_dir, scenarios_path)

        # 9) Alerts
        st.info("🔄 Step 9: Alert Generation...")
        alert_decider = AlertDecider()
        alerts_path = os.path.join(out_dir, "alerts.jsonl")
        alert_decider.run(out_dir, alerts_path)

        # 10) Ledger
        st.info("🔄 Step 10: Audit Trail...")
        audit_ledger = AuditLedger()
        ledger_path = os.path.join(out_dir, "ledger.jsonl")
        audit_ledger.run(out_dir, ledger_path)

        # Load results for demo
        alerts = []
        if os.path.exists(alerts_path):
            with open(alerts_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        alerts.append(json.loads(line.strip()))
                    except:
                        continue

    return {
        "articles": articles,
            "scores": [],  # Placeholder
            "blocks": [],  # Placeholder
            "scenarios": [],  # Placeholder
        "alerts": alerts
    }

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        return {
            "articles": [],
            "scores": [],
            "blocks": [],
            "scenarios": [],
            "alerts": []
        }

# -------------------------------------------------------------------
# Main UI
def main():
    st.set_page_config(
        page_title="🌍 NWW Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'bundle_dir' not in st.session_state:
        st.session_state.bundle_dir = "data/bundles/sample"
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

    # 사이드바
    render_sidebar()

    st.title("🌍 NWW - News War Watch")
    st.markdown("**위기 조기 탐지 및 분석 자동화 패키지**")

    # Tabs
    landing, overview, ingest_tab, scoring_tab, timeline_tab, blocks_tab, scenarios_tab, artifacts_tab, ledger_tab = st.tabs([
        "🦉 랜딩", "📊 Overview", "📥 Ingest", "🎯 Scoring",
        "⏰ Timeline", "🧱 Blocks", "📋 Scenarios", "📦 Artifacts", "📝 Ledger"
    ])

    with landing:
        render_landing_tab()
    with overview:
        render_overview_tab()
    with ingest_tab:
        render_ingest_tab()
    with scoring_tab:
        render_scoring_tab()
    with timeline_tab:
        render_timeline_tab()
    with blocks_tab:
        render_blocks_tab()
    with scenarios_tab:
        render_scenarios_tab()
    with artifacts_tab:
        render_artifacts_tab()
    with ledger_tab:
        render_ledger_tab()

# -------------------------------------------------------------------
# Sidebar
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.text_input("Bundle Directory", value=st.session_state.bundle_dir)

    st.sidebar.subheader("🔧 Settings")
    st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.05)
    st.sidebar.slider("EMA Alpha", 0.0, 1.0, 0.3, 0.05)
    st.sidebar.slider("Hysteresis", 0.0, 0.5, 0.1, 0.01)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run All Modules", type="primary"):
        st.success("🎉 Demo: All modules executed")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Status")
    modules = ["Ingest", "Normalize", "Analyze", "Gate", "Score IS", "Score DBN", "Score LLM", "Fusion", "Blocks", "Scenarios", "Alerts", "Ledger"]
    for m in modules:
        st.sidebar.info(f"⏸️ {m}")

# -------------------------------------------------------------------
# 랜딩 탭
def render_landing_tab():
    st.header("🦉 Crisis Overview")
    
    # Try to get alerts from session state first, then from file
    alerts = None
    if hasattr(st.session_state, 'alerts') and st.session_state.alerts:
        alerts = pd.DataFrame(st.session_state.alerts)
    else:
        alerts = get_alerts()
    
    if alerts is not None and not alerts.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌍 지역별 위기 현황")
            if "region" in alerts.columns:
                region_stats = alerts.groupby("region")["score"].count().reset_index(name="Active Alerts")
                fig1 = px.bar(region_stats, x="region", y="Active Alerts", color="region", title="지역별 위기 현황")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("⚠️ 데이터에 'region' 컬럼이 없습니다.")
        with col2:
            st.subheader("⚔️ 분야별 위기 현황")
            if "domain" in alerts.columns:
                domain_stats = alerts.groupby("domain")["score"].count().reset_index(name="Count")
                fig2 = px.pie(domain_stats, values="Count", names="domain", title="분야별 위기 현황")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("⚠️ 데이터에 'domain' 컬럼이 없습니다.")
    else:
        st.info("📝 아직 활성 알림이 없습니다. Ingest 탭에서 파이프라인을 실행하세요.")
        
        # Show sample data for demonstration
        st.subheader("📊 Sample Dashboard Preview")
        
        # Sample region data
        st.subheader("🌍 Crisis by Region (Sample)")
        sample_regions = pd.DataFrame({
            'region': ['asia', 'europe', 'americas', 'middle_east', 'africa'],
            'count': [45, 32, 28, 15, 8]
        })
        fig = px.bar(sample_regions, x='region', y='count', 
                    title="Sample Regional Crisis Distribution",
                    color='count', color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample domain data
        st.subheader("📂 Crisis by Domain (Sample)")
        sample_domains = pd.DataFrame({
            'domain': ['military', 'diplomacy', 'economy'],
            'count': [35, 28, 22]
        })
        fig = px.pie(sample_domains, names='domain', values='count',
                    title="Sample Domain Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Placeholders for other tabs
def render_overview_tab():
    st.header("📊 Overview")
    st.success("샘플: Overview 정상 동작")

def render_ingest_tab():
    st.header("📥 Data Ingestion & Pipeline")

    # URL input
    urls = st.text_area("Enter URLs (one per line):", height=150, 
                       placeholder="https://www.reuters.com/world/\nhttps://www.bbc.com/news\nhttps://www.cnn.com/world")

    # Sample URLs button
    if st.button("📋 Load Sample URLs"):
        sample_urls = """https://www.reuters.com/world/
https://www.bbc.com/news
https://www.cnn.com/world
https://www.ap.org/news"""
        st.session_state.sample_urls = sample_urls
        st.rerun()

    if "sample_urls" in st.session_state:
        st.text_area("Sample URLs loaded:", value=st.session_state.sample_urls, height=100, disabled=True)

    # Pipeline execution
    if st.button("🚀 Run Full Pipeline", type="primary"):
        if not urls.strip():
            st.warning("❗ URL을 입력해주세요.")
        else:
            url_list = [u.strip() for u in urls.splitlines() if u.strip()]
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 Starting pipeline...")
                progress_bar.progress(10)
                
                results = run_pipeline(url_list, out_dir=st.session_state.bundle_dir)
                
                progress_bar.progress(100)
                status_text.text("✅ Pipeline completed!")
                
            st.session_state.articles = results["articles"]
            st.session_state.alerts = results["alerts"]
            st.success("✅ 전체 파이프라인 실행 완료!")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                progress_bar.progress(0)
                status_text.text("❌ Pipeline failed")

    # Display results
    if hasattr(st.session_state, 'articles') and st.session_state.articles:
        st.subheader("📊 Processed Articles")
        df = pd.DataFrame(st.session_state.articles)
        
        # Show available columns
        available_cols = ["id", "title", "ts"]
        if "region" in df.columns:
            available_cols.append("region")
        if "domain" in df.columns:
            available_cols.append("domain")
        if "lang" in df.columns:
            available_cols.append("lang")
            
        st.dataframe(df[available_cols], use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            if "region" in df.columns:
                st.metric("Regions", df["region"].nunique())
        with col3:
            if "domain" in df.columns:
                st.metric("Domains", df["domain"].nunique())
    else:
        st.info("📝 No articles processed yet. Enter URLs and run the pipeline.")

def render_scoring_tab():
    st.header("🎯 Scoring")
    st.info("샘플: 점수 분석 탭 (구현 필요)")

def render_timeline_tab():
    st.header("⏰ Timeline")
    st.info("샘플: 시계열 분석 탭 (구현 필요)")

def render_blocks_tab():
    st.header("🧱 Blocks")
    st.info("샘플: EDS 블록 탭 (구현 필요)")

def render_scenarios_tab():
    st.header("📋 Scenarios")
    st.info("샘플: 시나리오 탭 (구현 필요)")

def render_artifacts_tab():
    st.header("📦 Artifacts")
    st.info("샘플: 내보내기 탭 (구현 필요)")

def render_ledger_tab():
    st.header("📝 Ledger")
    st.info("샘플: 감사 로그 탭 (구현 필요)")

# -------------------------------------------------------------------
# Helpers
def get_alerts():
    try:
        alerts_path = os.path.join(st.session_state.bundle_dir, "alerts.jsonl")
        if os.path.exists(alerts_path):
            data = []
            with open(alerts_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except:
                        continue
            if data:
                return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading alerts.jsonl: {e}")
    return None

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
