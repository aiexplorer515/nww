import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from datetime import datetime

st.set_page_config(page_title="🌍 NWW MVP Dashboard", layout="wide")

st.title("🌍 NWW MVP Dashboard (v1.2+Extended)")

DATA_DIR = "data/bundles/sample/"

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# -------------------
# Tabs
# -------------------
tabs = st.tabs([
    "Overview", "Ingest", "Scoring", "Timeline",
    "Blocks", "Events", "Scenarios", "Artifacts", "Ledger"
])

# -------------------
# Overview
# -------------------
with tabs[0]:
    st.header("Overview")
    articles = load_jsonl(os.path.join(DATA_DIR, "articles.jsonl"))
    scores = load_jsonl(os.path.join(DATA_DIR, "scores.jsonl"))
    st.metric("기사 수", len(articles))
    st.metric("스코어 샘플", len(scores))
    st.success("샘플 데이터 로딩 완료")

# -------------------
# Ingest
# -------------------
with tabs[1]:
    st.header("Ingested Articles")
    articles = load_jsonl(os.path.join(DATA_DIR, "articles.jsonl"))
    if articles:
        df = pd.DataFrame(articles)
        st.dataframe(df)
    else:
        st.warning("No articles.jsonl found")

# -------------------
# Scoring
# -------------------
with tabs[2]:
    st.header("Risk Scoring Distribution")
    scores = load_jsonl(os.path.join(DATA_DIR, "scores.jsonl"))
    if scores:
        df = pd.DataFrame(scores)
        fig = px.histogram(df, x="score", nbins=10, title="Risk Score Histogram")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No scores.jsonl found")

# -------------------
# Timeline
# -------------------
with tabs[3]:
    st.header("Timeline (Risk Scores)")
    events = load_jsonl(os.path.join(DATA_DIR, "event_blocks.jsonl"))
    if events:
        df = pd.DataFrame(events)
        if "risk_curve" in df.columns:
            for _, row in df.iterrows():
                curve = row["risk_curve"]
                ts = list(range(len(curve)))
                chart_df = pd.DataFrame({"t": ts, "risk": curve})
                st.subheader(f"Event {row['event_id']}")
                fig = px.line(chart_df, x="t", y="risk", markers=True, title="Risk Curve")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No event_blocks.jsonl found")

# -------------------
# Blocks
# -------------------
with tabs[4]:
    st.header("Crisis Blocks")
    blocks = load_jsonl(os.path.join(DATA_DIR, "block_hits.jsonl"))
    if blocks:
        for blk in blocks:
            st.json(blk)
    else:
        st.warning("No block_hits.jsonl found")

# -------------------
# Events
# -------------------
with tabs[5]:
    st.header("Event Aggregation")
    events = load_jsonl(os.path.join(DATA_DIR, "event_blocks.jsonl"))
    if events:
        for evt in events:
            st.subheader(evt["event_id"])
            st.write(f"Actors: {', '.join(evt.get('actors', []))}")
            st.write(f"Location: {evt.get('location')}")
            st.write(f"State: {evt.get('state')}")
            st.json(evt)
    else:
        st.warning("No event_blocks.jsonl found")

# -------------------
# Scenarios
# -------------------
with tabs[6]:
    st.header("Scenarios")
    scenarios = load_jsonl(os.path.join(DATA_DIR, "scenarios.jsonl"))
    if scenarios:
        for scn in scenarios:
            st.subheader(scn["title"])
            st.write(scn["description"])
            st.json(scn)
    else:
        st.warning("No scenarios.jsonl found")

# -------------------
# Artifacts
# -------------------
with tabs[7]:
    st.header("Artifacts / Reports")
    files = ["articles.jsonl", "scores.jsonl", "block_hits.jsonl", "event_blocks.jsonl", "scenarios.jsonl", "alerts.jsonl"]
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            st.download_button(label=f"Download {f}", data=open(path, "rb"), file_name=f)

# -------------------
# Ledger
# -------------------
with tabs[8]:
    st.header("Ledger (Audit Trail)")
    ledger_dir = os.path.join(DATA_DIR, "ledger")
    if os.path.exists(ledger_dir):
        for file in os.listdir(ledger_dir):
            st.subheader(file)
            with open(os.path.join(ledger_dir, file), encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    st.json(json.loads(line))
    else:
        st.warning("No ledger files found")
