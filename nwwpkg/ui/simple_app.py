"""
NWW Simple UI - Basic Streamlit dashboard without complex imports
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import yaml

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
        st.success("🎉 All modules completed! (Demo mode)")
    
    # Status display
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Status")
    
    # Demo status
    modules = ["Ingest", "Normalize", "Analyze", "Gate", "Score IS", "Score DBN", "Score LLM", 
               "Fusion", "Blocks", "Scenarios", "Alerts", "Ledger"]
    
    for module in modules:
        st.sidebar.success(f"✅ {module}")

def render_overview_tab():
    """Render overview tab with system status and key metrics."""
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles Processed", "1,247", "↗️ 12")
    
    with col2:
        st.metric("Active Alerts", "8", "↗️ 3")
    
    with col3:
        st.metric("Avg Score", "0.65", "↗️ 0.05")
    
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
    
    for step, description in pipeline_steps:
        st.success(f"✅ {step}: {description}")
    
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
        
        # Demo sources data
        sources_data = pd.DataFrame({
            'URL': [
                'https://www.reuters.com/world/',
                'https://www.bbc.com/news',
                'https://www.cnn.com/world',
                'https://www.ap.org/news'
            ],
            'Status': ['Active', 'Active', 'Active', 'Active'],
            'Last Update': ['2 min ago', '5 min ago', '1 min ago', '3 min ago'],
            'Success Rate': ['98%', '95%', '92%', '97%']
        })
        
        st.dataframe(sources_data, use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Controls")
        
        if st.button("🔄 Refresh Sources"):
            st.success("✅ Sources refreshed!")
        
        if st.button("▶️ Run Ingest"):
            st.success("✅ Ingestion completed!")
    
    # Ingestion statistics
    st.subheader("📊 Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sources", "4")
    
    with col2:
        st.metric("Success Rate", "95%", "↗️ 2%")
    
    with col3:
        st.metric("Last Update", "2 min ago")

def render_scoring_tab():
    """Render scoring tab for score analysis."""
    st.header("🎯 Scoring Analysis")
    
    # Score distribution
    st.subheader("📊 Score Distribution")
    
    # Demo scores data
    scores_data = pd.DataFrame({
        'score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 10,
        'stage': ['IS', 'DBN', 'LLM', 'FUSION'] * 22 + ['IS', 'DBN']
    })
    
    fig = px.histogram(scores_data, x='score', nbins=20, title="Score Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Score by stage
    st.subheader("📈 Scores by Stage")
    
    stage_scores = scores_data.groupby('stage')['score'].agg(['mean', 'std', 'count']).reset_index()
    st.dataframe(stage_scores, use_container_width=True)
    
    # Top scoring articles
    st.subheader("🏆 Top Scoring Articles")
    top_articles = pd.DataFrame({
        'ID': ['a1', 'a2', 'a3', 'a4', 'a5'],
        'Stage': ['FUSION', 'FUSION', 'FUSION', 'FUSION', 'FUSION'],
        'Score': [0.95, 0.89, 0.87, 0.84, 0.82]
    })
    st.dataframe(top_articles, use_container_width=True)

def render_timeline_tab():
    """Render timeline tab for temporal analysis."""
    st.header("⏰ Timeline Analysis")
    
    # Timeline visualization
    st.subheader("📅 Event Timeline")
    
    # Demo events data
    events_data = pd.DataFrame({
        'start_time': pd.date_range('2025-09-14 00:00', periods=10, freq='2H'),
        'end_time': pd.date_range('2025-09-14 02:00', periods=10, freq='2H'),
        'event_type': ['Military', 'Diplomatic', 'Economic', 'Military', 'Diplomatic'] * 2,
        'severity': ['High', 'Medium', 'Low', 'Critical', 'High'] * 2
    })
    
    fig = px.timeline(events_data, x_start='start_time', x_end='end_time', y='event_type',
                     color='severity', title="Event Timeline")
    st.plotly_chart(fig, use_container_width=True)

def render_blocks_tab():
    """Render blocks tab for EDS block analysis."""
    st.header("🧱 EDS Blocks")
    
    # Demo blocks data
    blocks_data = pd.DataFrame({
        'Block ID': ['B001', 'B002', 'B003', 'B004'],
        'Type': ['Military', 'Diplomatic', 'Economic', 'Military'],
        'Confidence': [0.95, 0.87, 0.76, 0.92],
        'Articles': [5, 3, 7, 4],
        'Status': ['Active', 'Active', 'Resolved', 'Active']
    })
    
    st.dataframe(blocks_data, use_container_width=True)

def render_scenarios_tab():
    """Render scenarios tab for scenario analysis."""
    st.header("📋 Scenarios")
    
    # Demo scenarios data
    scenarios_data = pd.DataFrame({
        'Scenario ID': ['S001', 'S002', 'S003'],
        'Name': ['Military Conflict', 'Diplomatic Crisis', 'Economic Warfare'],
        'Severity': ['High', 'Medium', 'Medium'],
        'Confidence': [0.89, 0.76, 0.68],
        'Status': ['Active', 'Monitoring', 'Resolved']
    })
    
    st.dataframe(scenarios_data, use_container_width=True)

def render_artifacts_tab():
    """Render artifacts tab for export functionality."""
    st.header("📦 Artifacts & Export")
    
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export JSON"):
            st.success("📄 JSON export completed!")
        
        if st.button("📋 Export Brief"):
            st.success("📋 Brief export completed!")
    
    with col2:
        if st.button("📦 Export ZIP"):
            st.success("📦 ZIP export completed!")
        
        if st.button("🔒 Export STIX"):
            st.success("🔒 STIX export completed!")
    
    # Export history
    st.subheader("📚 Export History")
    export_history = pd.DataFrame({
        'Timestamp': ['2025-09-14 10:30', '2025-09-14 09:15', '2025-09-14 08:00'],
        'Format': ['JSON', 'STIX', 'ZIP'],
        'Size': ['2.3 MB', '1.8 MB', '4.1 MB'],
        'Status': ['Completed', 'Completed', 'Completed']
    })
    st.dataframe(export_history, use_container_width=True)

def render_ledger_tab():
    """Render ledger tab for audit trail."""
    st.header("📝 Audit Ledger")
    
    # Demo ledger data
    ledger_data = pd.DataFrame({
        'Timestamp': ['2025-09-14 10:30:00', '2025-09-14 10:25:00', '2025-09-14 10:20:00'],
        'Step': ['Alerts', 'Scenarios', 'Blocks'],
        'Description': ['Alert generation completed', 'Scenario construction completed', 'Block matching completed'],
        'Status': ['Completed', 'Completed', 'Completed']
    })
    
    st.dataframe(ledger_data, use_container_width=True)

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

if __name__ == "__main__":
    main()



