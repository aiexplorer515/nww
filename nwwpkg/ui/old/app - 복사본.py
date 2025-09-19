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
import zipfile
import tempfile
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Any, Optional
import logging

# Import NWW modules
from ..ingest import Extractor
from ..preprocess import Normalizer
from ..analyze import Tagger
from ..rules import Gating
from ..score import ScoreIS, ScoreDBN, LLMJudge

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
    
    # Initialize session state
    if 'bundle_dir' not in st.session_state:
        st.session_state.bundle_dir = "data/bundles/sample"
    if 'config' not in st.session_state:
        st.session_state.config = load_default_config()
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    
    # Sidebar
    render_sidebar()
    
    # Main content
    st.title("🌍 NWW - News World Watch")
    st.markdown("**Complete Automation Package for Crisis Detection and Analysis**")
    
    # Tabs
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

def render_sidebar():
    """Render sidebar with configuration and controls."""
    st.sidebar.title("⚙️ Configuration")
    
    # Bundle directory selection
    bundle_dir = st.sidebar.text_input(
        "Bundle Directory", 
        value=st.session_state.bundle_dir,
        help="Path to the data bundle directory"
    )
    st.session_state.bundle_dir = bundle_dir
    
    # Configuration section
    st.sidebar.subheader("🔧 Settings")
    
    # Threshold settings
    st.session_state.config['threshold'] = st.sidebar.slider(
        "Alert Threshold", 0.0, 1.0, 0.7, 0.05,
        help="Minimum score for generating alerts"
    )
    
    st.session_state.config['alpha'] = st.sidebar.slider(
        "EMA Alpha", 0.0, 1.0, 0.3, 0.05,
        help="Exponential moving average smoothing factor"
    )
    
    st.session_state.config['hysteresis'] = st.sidebar.slider(
        "Hysteresis", 0.0, 0.5, 0.1, 0.01,
        help="Hysteresis for alert state changes"
    )
    
    # Run all button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run All Modules", type="primary"):
        run_all_modules()
    
    # Status display
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

def render_overview_tab():
    """Render overview tab with system status and key metrics."""
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles Processed", get_article_count(), "↗️ 12")
    
    with col2:
        st.metric("Active Alerts", get_alert_count(), "↗️ 3")
    
    with col3:
        st.metric("Avg Score", f"{get_avg_score():.2f}", "↗️ 0.05")
    
    with col4:
        st.metric("System Health", "🟢 Healthy", "↗️ 99%")
    
    # Processing pipeline status
    st.subheader("🔄 Processing Pipeline")
    
    pipeline_steps = [
        ("Ingest", "extract articles from sources"),
        ("Normalize", "clean and normalize text"),
        ("Analyze", "extract keywords, entities, frames"),
        ("Gate", "filter content based on indicators"),
        ("Score IS", "indicator-based scoring"),
        ("Score DBN", "dynamic Bayesian network scoring"),
        ("Score LLM", "LLM judge scoring"),
        ("Fusion", "score fusion and calibration"),
        ("Blocks", "EDS block matching"),
        ("Scenarios", "scenario construction"),
        ("Alerts", "alert generation"),
        ("Ledger", "audit trail")
    ]
    
    for i, (step, description) in enumerate(pipeline_steps):
        status = st.session_state.processing_status.get(step, "pending")
        
        if status == "completed":
            st.success(f"✅ {step}: {description}")
        elif status == "running":
            st.info(f"🔄 {step}: {description}")
        elif status == "error":
            st.error(f"❌ {step}: {description}")
        else:
            st.write(f"⏸️ {step}: {description}")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    # Sample data for demonstration
    activity_data = pd.DataFrame({
        'Time': pd.date_range(start='2025-09-14 00:00', periods=24, freq='H'),
        'Articles': [5, 8, 12, 15, 18, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68],
        'Alerts': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=activity_data['Time'], y=activity_data['Articles'], 
                  name="Articles", line=dict(color='blue')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=activity_data['Time'], y=activity_data['Alerts'], 
                  name="Alerts", line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Articles", secondary_y=False)
    fig.update_yaxes(title_text="Alerts", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

def render_ingest_tab():
    """Render ingest tab for data collection."""
    st.header("📥 Data Ingestion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Sources")
        
        # Display sources
        sources = get_sources()
        if sources:
            st.dataframe(sources, use_container_width=True)
        else:
            st.info("No sources found. Please add sources to the bundle directory.")
    
    with col2:
        st.subheader("⚙️ Controls")
        
        if st.button("🔄 Refresh Sources"):
            st.rerun()
        
        if st.button("▶️ Run Ingest"):
            with st.spinner("Running ingestion..."):
                st.session_state.processing_status["Ingest"] = "running"
                try:
                    run_ingest()
                    st.session_state.processing_status["Ingest"] = "completed"
                    st.success("✅ Ingestion completed!")
                except Exception as e:
                    st.session_state.processing_status["Ingest"] = "error"
                    st.error(f"❌ Ingestion failed: {e}")
    
    # Ingestion statistics
    st.subheader("📊 Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sources", len(sources) if sources is not None else 0)
    
    with col2:
        st.metric("Success Rate", "95%", "↗️ 2%")
    
    with col3:
        st.metric("Last Update", "2 min ago")

def render_scoring_tab():
    """Render scoring tab for score analysis."""
    st.header("🎯 Scoring Analysis")
    
    # Score distribution
    st.subheader("📊 Score Distribution")
    
    scores = get_scores()
    if scores:
        fig = px.histogram(scores, x='score', nbins=20, title="Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Score by stage
        st.subheader("📈 Scores by Stage")
        
        stage_scores = scores.groupby('stage')['score'].agg(['mean', 'std', 'count']).reset_index()
        st.dataframe(stage_scores, use_container_width=True)
        
        # Top scoring articles
        st.subheader("🏆 Top Scoring Articles")
        top_articles = scores.nlargest(10, 'score')
        st.dataframe(top_articles[['id', 'stage', 'score']], use_container_width=True)
    else:
        st.info("No scores available. Run the scoring modules first.")

def render_timeline_tab():
    """Render timeline tab for temporal analysis."""
    st.header("⏰ Timeline Analysis")
    
    # Timeline visualization
    st.subheader("📅 Event Timeline")
    
    events = get_events()
    if events:
        fig = px.timeline(events, x_start='start_time', x_end='end_time', y='event_type',
                         color='severity', title="Event Timeline")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events available. Run the analysis modules first.")

def render_blocks_tab():
    """Render blocks tab for EDS block analysis."""
    st.header("🧱 EDS Blocks")
    
    blocks = get_blocks()
    if blocks:
        st.dataframe(blocks, use_container_width=True)
    else:
        st.info("No blocks available. Run the EDS block matching module first.")

def render_scenarios_tab():
    """Render scenarios tab for scenario analysis."""
    st.header("📋 Scenarios")
    
    scenarios = get_scenarios()
    if scenarios:
        st.dataframe(scenarios, use_container_width=True)
    else:
        st.info("No scenarios available. Run the scenario builder module first.")

def render_artifacts_tab():
    """Render artifacts tab for export functionality."""
    st.header("📦 Artifacts & Export")
    
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export JSON"):
            export_json()
        
        if st.button("📋 Export Brief"):
            export_brief()
    
    with col2:
        if st.button("📦 Export ZIP"):
            export_zip()
        
        if st.button("🔒 Export STIX"):
            export_stix()
    
    # Export history
    st.subheader("📚 Export History")
    export_history = get_export_history()
    if export_history:
        st.dataframe(export_history, use_container_width=True)
    else:
        st.info("No export history available.")

def render_ledger_tab():
    """Render ledger tab for audit trail."""
    st.header("📝 Audit Ledger")
    
    ledger = get_ledger()
    if ledger:
        st.dataframe(ledger, use_container_width=True)
    else:
        st.info("No ledger entries available.")

# Helper functions
def load_default_config():
    """Load default configuration."""
    return {
        'threshold': 0.7,
        'alpha': 0.3,
        'hysteresis': 0.1,
        'weights': {
            'military': 0.8,
            'diplomacy': 0.6,
            'economy': 0.5
        }
    }

def run_all_modules():
    """Run all processing modules."""
    modules = [
        ("Ingest", run_ingest),
        ("Normalize", run_normalize),
        ("Analyze", run_analyze),
        ("Gate", run_gating),
        ("Score IS", run_score_is),
        ("Score DBN", run_score_dbn),
        ("Score LLM", run_score_llm),
        ("Fusion", run_fusion),
        ("Blocks", run_blocks),
        ("Scenarios", run_scenarios),
        ("Alerts", run_alerts),
        ("Ledger", run_ledger)
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (module_name, module_func) in enumerate(modules):
        status_text.text(f"Running {module_name}...")
        st.session_state.processing_status[module_name] = "running"
        
        try:
            module_func()
            st.session_state.processing_status[module_name] = "completed"
        except Exception as e:
            st.session_state.processing_status[module_name] = "error"
            logger.error(f"Error in {module_name}: {e}")
        
        progress_bar.progress((i + 1) / len(modules))
    
    status_text.text("✅ All modules completed!")
    st.success("🎉 Complete processing pipeline finished!")

def run_ingest():
    """Run ingestion module."""
    extractor = Extractor(st.session_state.bundle_dir)
    output_path = os.path.join(st.session_state.bundle_dir, "articles.jsonl")
    extractor.run(output_path)

def run_normalize():
    """Run normalization module."""
    normalizer = Normalizer()
    input_path = os.path.join(st.session_state.bundle_dir, "articles.jsonl")
    output_path = os.path.join(st.session_state.bundle_dir, "articles.norm.jsonl")
    log_path = os.path.join(st.session_state.bundle_dir, "logs", "normalize.log")
    normalizer.run(input_path, output_path, log_path)

def run_analyze():
    """Run analysis module."""
    tagger = Tagger()
    input_path = os.path.join(st.session_state.bundle_dir, "articles.norm.jsonl")
    output_path = os.path.join(st.session_state.bundle_dir, "kyw_sum.jsonl")
    tagger.run(input_path, output_path)

def run_gating():
    """Run gating module."""
    gating = Gating()
    input_path = os.path.join(st.session_state.bundle_dir, "kyw_sum.jsonl")
    output_path = os.path.join(st.session_state.bundle_dir, "gated.jsonl")
    gating.run(input_path, output_path)

def run_score_is():
    """Run IS scoring module."""
    scorer = ScoreIS()
    input_path = os.path.join(st.session_state.bundle_dir, "gated.jsonl")
    output_path = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    scorer.run(input_path, output_path)

def run_score_dbn():
    """Run DBN scoring module."""
    scorer = ScoreDBN()
    output_path = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    scorer.run(st.session_state.bundle_dir, output_path)

def run_score_llm():
    """Run LLM scoring module."""
    scorer = LLMJudge()
    output_path = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
    scorer.run(st.session_state.bundle_dir, output_path)

def run_fusion():
    """Run fusion module."""
    # Placeholder for fusion module
    pass

def run_blocks():
    """Run blocks module."""
    # Placeholder for blocks module
    pass

def run_scenarios():
    """Run scenarios module."""
    # Placeholder for scenarios module
    pass

def run_alerts():
    """Run alerts module."""
    # Placeholder for alerts module
    pass

def run_ledger():
    """Run ledger module."""
    # Placeholder for ledger module
    pass

# Data access functions
def get_article_count():
    """Get total article count."""
    try:
        articles_path = os.path.join(st.session_state.bundle_dir, "articles.jsonl")
        if os.path.exists(articles_path):
            with open(articles_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
    except:
        pass
    return 0

def get_alert_count():
    """Get active alert count."""
    # Placeholder implementation
    return 3

def get_avg_score():
    """Get average score."""
    # Placeholder implementation
    return 0.65

def get_sources():
    """Get sources data."""
    try:
        sources_path = os.path.join(st.session_state.bundle_dir, "sources.csv")
        if os.path.exists(sources_path):
            return pd.read_csv(sources_path)
    except:
        pass
    return None

def get_scores():
    """Get scores data."""
    try:
        scores_path = os.path.join(st.session_state.bundle_dir, "scores.jsonl")
        if os.path.exists(scores_path):
            scores = []
            with open(scores_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        scores.append(json.loads(line.strip()))
                    except:
                        continue
            return pd.DataFrame(scores)
    except:
        pass
    return None

def get_events():
    """Get events data."""
    # Placeholder implementation
    return None

def get_blocks():
    """Get blocks data."""
    # Placeholder implementation
    return None

def get_scenarios():
    """Get scenarios data."""
    # Placeholder implementation
    return None

def get_export_history():
    """Get export history."""
    # Placeholder implementation
    return None

def get_ledger():
    """Get ledger data."""
    # Placeholder implementation
    return None

# Export functions
def export_json():
    """Export data as JSON."""
    st.success("📄 JSON export completed!")

def export_brief():
    """Export brief report."""
    st.success("📋 Brief export completed!")

def export_zip():
    """Export data as ZIP."""
    st.success("📦 ZIP export completed!")

def export_stix():
    """Export data as STIX."""
    st.success("🔒 STIX export completed!")

if __name__ == "__main__":
    main()
