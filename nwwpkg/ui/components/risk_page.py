from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8", errors="ignore")) if p.exists() else {}

def _read_jsonl_df(p: Path):
    rows=[]
    if p.exists():
        for L in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if L.strip():
                rows.append(json.loads(L))
    return pd.DataFrame(rows)

def render_risk_page(root: Path):
    st.header("🔎 Risk Analysis")
    topics = _read_json(root/"risk.topics.json").get("topics", [])
    dfE = _read_jsonl_df(root/"risk.events.jsonl")

    if dfE.empty and not topics:
        st.info("risk 데이터가 없습니다. tools/risk_engine.py 실행을 먼저 하세요.")
        return

    # 토픽 테이블
    if topics:
        st.subheader("주제별 누적")
        dfT = pd.DataFrame(topics)
        st.dataframe(dfT, use_container_width=True)
        st.bar_chart(dfT.set_index("topic")[["avg"]])

    # 필터 + 사건 테이블
    st.subheader("사건 상세")
    if not dfE.empty:
        cols = st.columns(4)
        topic_sel = cols[0].multiselect("토픽", sorted(dfE["topic"].unique()), [])
        lvl_sel   = cols[1].multiselect("레벨", ["warn","alert","crisis"], ["alert","crisis"])
        min_s     = cols[2].slider("최소 점수", 0.0, 1.0, 0.4, 0.05)
        search    = cols[3].text_input("제목 검색")

        q = dfE.copy()
        if topic_sel: q = q[q["topic"].isin(topic_sel)]
        if lvl_sel:   q = q[q["level"].isin(lvl_sel)]
        q = q[q["score"]>=min_s]
        if search:    q = q[q["title"].str.contains(search, case=False, na=False)]
        q = q.sort_values(["score"], ascending=False)

        st.dataframe(q, use_container_width=True)
    else:
        st.info("사건 데이터가 없습니다.")
