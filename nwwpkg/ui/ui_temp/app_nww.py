import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os, subprocess
from datetime import datetime

# ---------- 유틸 ----------
def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def count_alerts(alerts):
    return sum(1 for a in alerts if a.get("level","ALERT")=="ALERT")

# ---------- 메인 ----------
def main():
    st.set_page_config(
        page_title="🌍 NWW Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if "bundle_dir" not in st.session_state:
        st.session_state.bundle_dir = "data/bundles/sample"

    render_sidebar()
    
    st.title("🌍 NWW - News World Watch")
    st.caption("Complete Automation Package for Crisis Detection and Analysis")
    
    tabs = st.tabs([
        "📊 Overview","📥 Ingest","🎯 Scoring","⏰ Timeline",
        "🧱 Blocks","📋 Scenarios","📦 Artifacts","📝 Ledger"
    ])
    
    with tabs[0]: render_overview()
    with tabs[1]: render_ingest()
    with tabs[2]: render_scoring()
    with tabs[3]: render_timeline()
    with tabs[4]: render_blocks()
    with tabs[5]: render_scenarios()
    with tabs[6]: render_artifacts()
    with tabs[7]: render_ledger()

# ---------- Sidebar ----------
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    st.session_state.bundle_dir = st.sidebar.text_input(
        "Bundle Directory", st.session_state.bundle_dir
    )
    
    st.sidebar.subheader("🔧 Settings")
    st.sidebar.slider("Alert Threshold",0.0,1.0,0.7,0.05)
    st.sidebar.slider("EMA Alpha",0.0,1.0,0.3,0.05)
    st.sidebar.slider("Hysteresis",0.0,0.5,0.1,0.01)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run All Modules"):
        try:
            subprocess.run(["python","run_all.py",st.session_state.bundle_dir],check=True)
            st.sidebar.success("✅ Pipeline executed")
        except Exception as e:
            st.sidebar.error(f"Run failed: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Status")
    for m in ["Ingest","Normalize","Analyze","Gate","IS","DBN","LLM","Fusion","Blocks","Scenarios","Alerts","Ledger"]:
        st.sidebar.success(f"✅ {m}")

# ---------- Tabs ----------
def render_overview():
    st.header("📊 System Overview")
    articles = load_jsonl(os.path.join(st.session_state.bundle_dir,"articles.jsonl"))
    scores   = load_jsonl(os.path.join(st.session_state.bundle_dir,"scores.jsonl"))
    alerts   = load_jsonl(os.path.join(st.session_state.bundle_dir,"alerts.jsonl"))
    
    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Articles",len(articles))
    with col2: st.metric("Alerts",count_alerts(alerts))
    with col3: 
        avg = round(pd.DataFrame(scores)["score"].mean(),2) if scores else 0
        st.metric("Avg Score",avg)
    with col4: st.metric("System Health","🟢 Healthy" if avg>0.5 else "🟠 Warning")
    
    # Pipeline steps
    st.subheader("Pipeline Status")
    for step in ["Ingest","Normalize","Analyze","Gate","IS","DBN","LLM","Fusion","Blocks","Scenarios","Alerts","Ledger"]:
        st.success(f"✅ {step} completed")

def render_ingest():
    st.header("📥 Ingested Articles")
    data = load_jsonl(os.path.join(st.session_state.bundle_dir,"articles.jsonl"))
    if data: st.dataframe(pd.DataFrame(data))
    else: st.warning("No articles.jsonl found")

def render_scoring():
    st.header("🎯 Scoring Analysis")
    scores = load_jsonl(os.path.join(st.session_state.bundle_dir,"scores.jsonl"))
    if not scores: return st.warning("No scores.jsonl")
    df = pd.DataFrame(scores)
    st.plotly_chart(px.histogram(df,x="score",nbins=20),use_container_width=True)
    st.dataframe(df.groupby("stage")["score"].describe())

def render_timeline():
    st.header("⏰ Timeline")
    events = load_jsonl(os.path.join(st.session_state.bundle_dir,"event_blocks.jsonl"))
    if not events: return st.warning("No event_blocks.jsonl")
    for evt in events:
        curve = evt.get("risk_curve",[])
        st.line_chart(curve)

def render_blocks():
    st.header("🧱 Blocks")
    blocks = load_jsonl(os.path.join(st.session_state.bundle_dir,"block_hits.jsonl"))
    for blk in blocks: st.json(blk)

def render_scenarios():
    st.header("📋 Scenarios")
    scenarios = load_jsonl(os.path.join(st.session_state.bundle_dir,"scenarios.jsonl"))
    for scn in scenarios: st.json(scn)

def render_artifacts():
    st.header("📦 Artifacts")
    for f in ["articles.jsonl","scores.jsonl","block_hits.jsonl","event_blocks.jsonl","scenarios.jsonl","alerts.jsonl"]:
        path = os.path.join(st.session_state.bundle_dir,f)
        if os.path.exists(path):
            st.download_button(f"Download {f}",open(path,"rb"),file_name=f)

def render_ledger():
    st.header("📝 Ledger")
    ledger_dir = os.path.join(st.session_state.bundle_dir,"ledger")
    if not os.path.exists(ledger_dir): return st.warning("No ledger dir")
    for file in os.listdir(ledger_dir):
        st.subheader(file)
        for line in open(os.path.join(ledger_dir,file),encoding="utf-8"):
            st.json(json.loads(line))

if __name__=="__main__":
    main()
