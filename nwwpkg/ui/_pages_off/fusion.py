# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from pathlib import Path
import streamlit as st
import pandas as pd

def _coerce_root(root: Path | str | None) -> Path:
    if isinstance(root, Path):
        return root
    base = Path(os.getenv("NWW_DATA_HOME", "data"))
    if isinstance(root, str) and root:
        return base / root
    return base / os.getenv("NWW_BUNDLE", "b01")

def _read_jsonl(p: Path):
    rows=[]
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s: rows.append(json.loads(s))
    return rows

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8", errors="ignore")) if p.exists() else {}

def page_fusion(root: Path | str = None):
    """ROOT 기반 Fusion 페이지 (frames/risk/alerts/eventblocks 등 융합 뷰)"""
    root = _coerce_root(root)
    if not root.exists():
        st.error(f"[Fusion] 데이터 루트 없음: {root}")
        return

    st.header("🧬 Fusion")
    st.caption(f"root: {root}")

    # 입력들(있으면 읽고, 없으면 빈 값)
    risk_topics = _read_json(root/"risk.topics.json").get("topics", [])
    alerts      = _read_jsonl(root/"alerts.jsonl")
    blocks      = _read_jsonl(root/"eventblocks.jsonl")
    scores      = _read_jsonl(root/"scores.jsonl")

    # KPI
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk 토픽", len(risk_topics))
    c2.metric("Alerts", len(alerts))
    c3.metric("EventBlocks", len(blocks))
    c4.metric("Scores", len(scores))

    # 간단한 융합 테이블(예시)
    dfA = pd.DataFrame(alerts)
    dfB = pd.DataFrame(blocks)
    if not dfA.empty:
        st.subheader("Alerts 상위(점수순)")
        st.dataframe(dfA.sort_values("score", ascending=False).head(50), use_container_width=True, height=300)
    if not dfB.empty:
        st.subheader("EventBlocks 요약")
        st.dataframe(pd.DataFrame(blocks), use_container_width=True, height=300)

    st.info("※ 실제 Fusion 로직(조인/랭킹/하이라이트)은 프로젝트 룰에 맞게 이어서 구현하면 됩니다.")
