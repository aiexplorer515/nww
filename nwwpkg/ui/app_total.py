import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import json
from datetime import datetime, timedelta
import time

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with correct paths
from nwwpkg.ingest import Extractor
from nwwpkg.preprocess import Normalizer
from nwwpkg.analyze import Tagger
from nwwpkg.rules import Gating
from nwwpkg.score import ScoreIS, ScoreDBN, LLMJudge
from nwwpkg.fusion import FusionCalibration
from nwwpkg.eds import EDSBlockMatcher
from nwwpkg.scenario import ScenarioBuilder
from nwwpkg.decider import AlertDecider
from nwwpkg.ledger import AuditLedger
from nwwpkg.eventblock import EventBlockAggregator

# Page configuration
st.set_page_config(
    page_title="NWW - News Watch & Warning System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = {
        'articles': [],
        'normalized': [],
        'tagged': [],
        'gated': [],
        'scored': [],
        'fused': [],
        'blocks': [],
        'scenarios': [],
        'alerts': []
    }

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "Ready"

def run_pipeline(urls=None, sample_data=None):
    """Run the complete NWW pipeline"""
    try:
        st.session_state.processing_status = "Running"
        
        # Initialize components
        extractor = Extractor()
        normalizer = Normalizer()
        tagger = Tagger()
        gating = Gating()
        score_is = ScoreIS()
        score_dbn = ScoreDBN()
        llm_judge = LLMJudge()
        fusion = FusionCalibration()
        eds_matcher = EDSBlockMatcher()
        scenario_builder = ScenarioBuilder()
        alert_decider = AlertDecider()
        audit_ledger = AuditLedger()
        event_aggregator = EventBlockAggregator()
        
        # Step 1: Ingest
        if urls:
            articles = extractor.extract_from_urls(urls)
        elif sample_data:
            articles = sample_data
        else:
            articles = []
        
        st.session_state.pipeline_data['articles'] = articles
        
        # Step 2: Normalize
        normalized = []
        for article in articles:
            normalized_article = normalizer.normalize(article)
            normalized.append(normalized_article)
        
        st.session_state.pipeline_data['normalized'] = normalized
        
        # Step 3: Tag
        tagged = []
        for article in normalized:
            tagged_article = tagger.tag(article)
            tagged.append(tagged_article)
        
        st.session_state.pipeline_data['tagged'] = tagged
        
        # Step 4: Gate
        gated = []
        for article in tagged:
            if gating.should_process(article):
                gated.append(article)
        
        st.session_state.pipeline_data['gated'] = gated
        
        # Step 5: Score
        scored = []
        for article in gated:
            is_score = score_is.score(article)
            dbn_score = score_dbn.score(article)
            llm_score = llm_judge.score(article)
            
            article['scores'] = {
                'is': is_score,
                'dbn': dbn_score,
                'llm': llm_score
            }
            scored.append(article)
        
        st.session_state.pipeline_data['scored'] = scored
        
        # Step 6: Fusion
        fused = []
        for article in scored:
            fused_score = fusion.fuse(article['scores'])
            article['fused_score'] = fused_score
            fused.append(article)
        
        st.session_state.pipeline_data['fused'] = fused
        
        # Step 7: EDS Block Matching
        blocks = eds_matcher.match_blocks(fused)
        st.session_state.pipeline_data['blocks'] = blocks
        
        # Step 8: Scenario Building
        scenarios = scenario_builder.build_scenarios(blocks)
        st.session_state.pipeline_data['scenarios'] = scenarios
        
        # Step 9: Alert Decision
        alerts = alert_decider.decide_alerts(scenarios)
        st.session_state.pipeline_data['alerts'] = alerts
        
        # Step 10: Audit Ledger
        audit_ledger.record_processing({
            'timestamp': datetime.now(),
            'articles_processed': len(articles),
            'alerts_generated': len(alerts)
        })
        
        st.session_state.processing_status = "Completed"
        return True
        
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        st.session_state.processing_status = "Error"
        return False

def render_landing_tab():
    """Render the landing/home tab"""
    st.header("🏠 NWW Dashboard")
    
    # Status indicator
    status_color = {
        "Ready": "🟢",
        "Running": "🟡", 
        "Completed": "🟢",
        "Error": "🔴"
    }
    
    st.metric(
        label="System Status",
        value=f"{status_color.get(st.session_state.processing_status, '⚪')} {st.session_state.processing_status}"
    )
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles", len(st.session_state.pipeline_data['articles']))
    
    with col2:
        st.metric("Alerts", len(st.session_state.pipeline_data['alerts']))
    
    with col3:
        st.metric("Scenarios", len(st.session_state.pipeline_data['scenarios']))
    
    with col4:
        st.metric("Blocks", len(st.session_state.pipeline_data['blocks']))
    
    # Sample data display
    if st.session_state.pipeline_data['alerts']:
        st.subheader("📊 Recent Alerts")
        alerts_df = pd.DataFrame(st.session_state.pipeline_data['alerts'])
        
        if not alerts_df.empty:
            # Region distribution
            if "region" in alerts_df.columns and not alerts_df["region"].empty:
                region_counts = alerts_df["region"].value_counts()
                fig_region = px.bar(
                    x=region_counts.index, 
                    y=region_counts.values,
                    title="Alerts by Region",
                    labels={'x': 'Region', 'y': 'Count'}
                )
                st.plotly_chart(fig_region, use_container_width=True)
            
            # Domain distribution
            if "domain" in alerts_df.columns and not alerts_df["domain"].empty:
                domain_counts = alerts_df["domain"].value_counts().head(10)
                fig_domain = px.bar(
                    x=domain_counts.index, 
                    y=domain_counts.values,
                    title="Top 10 Alert Sources",
                    labels={'x': 'Domain', 'y': 'Count'}
                )
                st.plotly_chart(fig_domain, use_container_width=True)
            
            # Alert timeline
            if "timestamp" in alerts_df.columns:
                alerts_df['date'] = pd.to_datetime(alerts_df['timestamp']).dt.date
                daily_counts = alerts_df.groupby('date').size().reset_index(name='count')
                fig_timeline = px.line(
                    daily_counts, 
                    x='date', 
                    y='count',
                    title="Daily Alert Trends"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No alerts available. Run the pipeline to generate alerts.")
        
        # Show sample data structure
        st.subheader("📋 Sample Data Structure")
        sample_data = {
            "title": "Sample News Article",
            "body": "This is a sample news article for demonstration purposes.",
            "url": "https://example.com/sample-article",
            "timestamp": datetime.now().isoformat(),
            "domain": "example.com",
            "region": "Global"
        }
        st.json(sample_data)

def render_ingest_tab():
    """Render the ingest tab"""
    st.header("📥 Data Ingestion")
    
    # URL input
    st.subheader("Add News Sources")
    urls_input = st.text_area(
        "Enter URLs (one per line):",
        placeholder="https://example.com/article1\nhttps://example.com/article2",
        height=100
    )
    
    # Sample data option
    st.subheader("Or Use Sample Data")
    if st.button("Load Sample Data"):
        sample_data = [
            {
                "title": "Breaking: Major Event Occurs",
                "body": "A significant event has occurred that requires immediate attention and analysis.",
                "url": "https://example.com/breaking-news",
                "timestamp": datetime.now().isoformat(),
                "domain": "example.com",
                "region": "Global"
            },
            {
                "title": "Economic Update",
                "body": "Latest economic indicators show positive trends in the market.",
                "url": "https://example.com/economic-update",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "domain": "example.com",
                "region": "Global"
            }
        ]
        st.session_state.pipeline_data['articles'] = sample_data
        st.success(f"Loaded {len(sample_data)} sample articles")
    
    # Process button
    if st.button("🚀 Run Pipeline", type="primary"):
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()] if urls_input else None
        
        with st.spinner("Processing pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress
            for i in range(10):
                progress_bar.progress((i + 1) * 10)
                status_text.text(f"Processing step {i + 1}/10...")
                time.sleep(0.1)
            
            success = run_pipeline(urls=urls)
            
            if success:
                st.success("Pipeline completed successfully!")
                st.rerun()
            else:
                st.error("Pipeline failed. Check the error messages above.")

def render_analysis_tab():
    """Render the analysis tab"""
    st.header("🔍 Analysis Results")
    
    if not st.session_state.pipeline_data['tagged']:
        st.info("No analysis data available. Run the pipeline first.")
        return
    
    # Show tagged articles
    st.subheader("Tagged Articles")
    tagged_df = pd.DataFrame(st.session_state.pipeline_data['tagged'])
    st.dataframe(tagged_df, use_container_width=True)
    
    # Show scoring results
    if st.session_state.pipeline_data['scored']:
        st.subheader("Scoring Results")
        scored_data = st.session_state.pipeline_data['scored']
        
        # Create scores dataframe
        scores_data = []
        for article in scored_data:
            if 'scores' in article:
                scores_data.append({
                    'title': article.get('title', 'Unknown'),
                    'is_score': article['scores'].get('is', 0),
                    'dbn_score': article['scores'].get('dbn', 0),
                    'llm_score': article['scores'].get('llm', 0),
                    'fused_score': article.get('fused_score', 0)
                })
        
        if scores_data:
            scores_df = pd.DataFrame(scores_data)
            st.dataframe(scores_df, use_container_width=True)
            
            # Score distribution
            fig_scores = px.histogram(
                scores_df, 
                x=['is_score', 'dbn_score', 'llm_score', 'fused_score'],
                title="Score Distribution",
                barmode='overlay'
            )
            st.plotly_chart(fig_scores, use_container_width=True)

def render_alerts_tab():
    """Render the alerts tab"""
    st.header("🚨 Alert Management")
    
    if not st.session_state.pipeline_data['alerts']:
        st.info("No alerts available. Run the pipeline to generate alerts.")
        return
    
    alerts_df = pd.DataFrame(st.session_state.pipeline_data['alerts'])
    
    # Alert summary
    st.subheader("Alert Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Alerts", len(alerts_df))
    
    with col2:
        if 'severity' in alerts_df.columns:
            high_severity = len(alerts_df[alerts_df['severity'] == 'high'])
            st.metric("High Severity", high_severity)
    
    # Alert details
    st.subheader("Alert Details")
    st.dataframe(alerts_df, use_container_width=True)
    
    # Alert actions
    st.subheader("Alert Actions")
    if st.button("Export Alerts"):
        csv = alerts_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("📰 NWW System")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Home", "📥 Ingest", "🔍 Analysis", "🚨 Alerts"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("NWW - News Watch & Warning System")
    
    # Main content
    if page == "🏠 Home":
        render_landing_tab()
    elif page == "📥 Ingest":
        render_ingest_tab()
    elif page == "🔍 Analysis":
        render_analysis_tab()
    elif page == "🚨 Alerts":
        render_alerts_tab()

if __name__ == "__main__":
    main()