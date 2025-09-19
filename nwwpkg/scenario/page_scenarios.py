# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time
from pathlib import Path
import streamlit as st
import pandas as pd
from llm_utils import generate_scenario_with_llm   # ← 분리된 LLM 유틸 불러오기

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

    # 🔽 모델 선택 옵션
    model_name = st.selectbox(
        "LLM 모델 선택",
        options=["gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"],
        index=0
    )

    # =============== 개별 모드 ===============
    st.subheader("개별 블록 기반 시나리오")
    pick = st.selectbox("시나리오 시드로 사용할 블록", options=dfB["block_id"].tolist())
    sel = dfB[dfB["block_id"]==pick].iloc[0].to_dict()

    events_path = root/"events.jsonl"
    events = _read_jsonl(events_path)
    dfE = pd.DataFrame(events) if events else pd.DataFrame()
    top_events = dfE[dfE.get("block")==pick]["normalized"].head(3).tolist() if not dfE.empty else []

    title = st.text_input("제목", value=f"[Scenario] {pick} · {sel.get('rep_title','')[:40]}")
    summary = st.text_area("개요(요약)", value=f"- 기간: {sel.get('start','')} ~ {sel.get('end','')}\n- 규모: {sel.get('count',0)}건\n- 키워드: {', '.join(sel.get('top_keywords', []))}")

    if st.button("LLM 자동 생성 (단일)"):
        auto_title, auto_summary = generate_scenario_with_llm(sel, top_events, model_name=model_name)
        st.session_state["scenario_title"] = auto_title
        st.session_state["scenario_summary"] = auto_summary
        st.session_state["scenario_model"] = model_name
        st.success(f"LLM({model_name})이 시나리오를 생성했습니다.")
    
    if "scenario_title" in st.session_state:
        title = st.text_input("제목 (자동 생성)", value=st.session_state["scenario_title"])
    if "scenario_summary" in st.session_state:
        summary = st.text_area("개요(자동 생성)", value=st.session_state["scenario_summary"])

    if st.button("시나리오 저장 (단일)"):
        outp = root/"scenarios.json"
        curr = json.loads(outp.read_text(encoding="utf-8", errors="ignore")) if outp.exists() else {"items":[]}
        curr["items"].append({
            "block_id": pick,
            "title": title,
            "summary": summary,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "LLM" if "scenario_model" in st.session_state else "manual",
            "model": st.session_state.get("scenario_model", "manual")
        })
        _write_json(outp, curr)
        st.success(f"저장됨: {outp}")

    # =============== 배치 모드 ===============
    st.subheader("🔄 여러 블록 자동 시나리오 생성 (Batch Mode)")

    if st.button("LLM 자동 생성 (여러 블록)"):
        outp = root/"scenarios.json"
        curr = json.loads(outp.read_text(encoding="utf-8", errors="ignore")) if outp.exists() else {"items":[]}

        for _, row in dfB.iterrows():
            blk = row.to_dict()
            blk_id = blk["block_id"]
            top_events_blk = dfE[dfE.get("block")==blk_id]["normalized"].head(3).tolist() if not dfE.empty else []

            auto_title, auto_summary = generate_scenario_with_llm(blk, top_events_blk, model_name=model_name)
            curr["items"].append({
                "block_id": blk_id,
                "title": auto_title,
                "summary": auto_summary,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "LLM-batch",
                "model": model_name
            })
            st.write(f"✅ {blk_id} → {auto_title}")

        _write_json(outp, curr)
        st.success(f"모든 블록({len(dfB)})에 대해 시나리오 자동 생성 완료.")
