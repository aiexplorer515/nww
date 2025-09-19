# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
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

def _write_jsonl(p: Path, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

def page_blocks(root: Path | str = None):
    """ROOT 기반 Blocks 관리(읽기 중심; 필요 시 편집 기능 추가)"""
    root = _coerce_root(root)
    if not root.exists():
        st.error(f"[Blocks] 데이터 루트 없음: {root}"); return

    st.header("🧱 Blocks")
    st.caption(f"root: {root}")

    blocks = _read_jsonl(root/"eventblocks.jsonl")
    timeline = json.loads((root/"eventblocks.timeline.json").read_text(encoding="utf-8", errors="ignore")) \
               if (root/"eventblocks.timeline.json").exists() else {"members":[]}
    members = timeline.get("members", [])

    dfB = pd.DataFrame(blocks)
    dfM = pd.DataFrame(members)

    c1,c2 = st.columns(2)
    c1.metric("블록 수", len(dfB))
    c2.metric("멤버 수", len(dfM))

    st.subheader("블록 목록")
    st.dataframe(dfB, use_container_width=True, height=300)

    if not dfB.empty:
        pick = st.selectbox("블록 선택", options=dfB["block_id"].tolist())
        sub = dfM[dfM["block_id"]==pick].copy()
        st.subheader("블록 멤버")
        st.dataframe(sub[["dt","alert_level","score","title","id"]], use_container_width=True, height=320)

    # (옵션) 편집/삭제 기능은 프로젝트 룰에 맞게 추가
    st.info("※ 편집(merge/split/relabel)은 요구사항에 맞춰 후속 구현.")
