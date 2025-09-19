from __future__ import annotations
import json, os, time
from pathlib import Path
import pandas as pd
import streamlit as st

def _read_json(p: Path):
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))

def _read_jsonl(p: Path):
    rows=[]; 
    if p.exists():
        for L in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if L.strip(): rows.append(json.loads(L))
    return rows

def render_risk_snapshot(root: Path):
    st.subheader("⚠️ Risk Snapshot")
    topics = _read_json(root/"risk.topics.json").get("topics", [])
    events = _read_jsonl(root/"risk.events.jsonl")

    # KPI
    topN = topics[:5]
    n_events = len(events)
    n_topics = len({t["topic"] for t in topics})
    n_alerts = sum(1 for e in events if e.get("level") in ("alert","crisis"))

    c1,c2,c3 = st.columns(3)
    c1.metric("누적 토픽 수", n_topics)
    c2.metric("사건 수", n_events)
    c3.metric("경보(alert+) 수", n_alerts)

    # Top-N 바차트
    if topN:
        df = pd.DataFrame(topN)[["topic","avg"]].set_index("topic")
        st.bar_chart(df, height=180)
    else:
        st.info("위험 토픽 데이터가 없습니다. risk_engine을 먼저 실행하세요.")

    # 최근 경보 feed
    st.write("#### 최신 경보")
    feed = [e for e in events if e.get("level") in ("alert","crisis")]
    feed.sort(key=lambda x: x.get("score",0), reverse=True)
    for e in feed[:8]:
        st.write(f"- **[{e['level']}]** `{e['topic']}` · s={e['score']:.2f} · {e.get('title','')}")
    st.caption(f"source: {root/'risk.topics.json'}, {root/'risk.events.jsonl'}")
