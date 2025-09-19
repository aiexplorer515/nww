# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time
from pathlib import Path
import streamlit as st
import pandas as pd

def _coerce_root(root: Path | str | None) -> Path:
    if isinstance(root, Path): return root
    base = Path(os.getenv("NWW_DATA_HOME","data"))
    if isinstance(root, str) and root: return base / root
    return base / os.getenv("NWW_BUNDLE","b01")

def _read_jsonl(p: Path):
    rows=[]; 
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s: rows.append(json.loads(s))
    return rows

def _write_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def page_scenarios(root: Path | str = None):
    """ROOT 기반 Scenario 초안 생성/관리(시드: EventBlocks)"""
    root = _coerce_root(root)
    if not root.exists():
        st.error(f"[Scenarios] 데이터 루트 없음: {root}"); return

    st.header("🗺️ Scenarios")
    st.caption(f"root: {root}")

    blocks = _read_jsonl(root/"eventblocks.jsonl")
    dfB = pd.DataFrame(blocks)

    if dfB.empty:
        st.info("EventBlocks가 없습니다. 먼저 Alerts → EventBlocks를 생성하세요.")
        return

    pick = st.selectbox("시나리오 시드로 사용할 블록", options=dfB["block_id"].tolist())
    sel = dfB[dfB["block_id"]==pick].iloc[0].to_dict()

    st.subheader("시나리오 초안")
    title = st.text_input("제목", value=f"[Scenario] {pick} · {sel.get('rep_title','')[:40]}")
    summary = st.text_area("개요(요약)", value=f"- 기간: {sel.get('start','')} ~ {sel.get('end','')}\n- 규모: {sel.get('count',0)}건\n- 키워드: {', '.join(sel.get('top_keywords', []))}")

    if st.button("시나리오 저장"):
        outp = root/"scenarios.json"
        curr = json.loads(outp.read_text(encoding="utf-8", errors="ignore")) if outp.exists() else {"items":[]}
        curr["items"].append({
            "block_id": pick,
            "title": title,
            "summary": summary,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        _write_json(outp, curr)
        st.success(f"저장됨: {outp}")
