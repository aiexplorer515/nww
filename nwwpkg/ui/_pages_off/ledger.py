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
    rows=[]
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s: rows.append(json.loads(s))
    return rows

def page_ledger(root: Path | str = None):
    """ROOT 기반 Ledger(감사/로깅) 뷰"""
    root = _coerce_root(root)
    if not root.exists():
        st.error(f"[Ledger] 데이터 루트 없음: {root}"); return

    st.header("📒 Ledger")
    st.caption(f"root: {root}")

    # 존재하는 산출물 개요
    files = [
        "clean.jsonl","frames.jsonl","frames_llm.jsonl","frames.llm.jsonl",
        "risk.topics.json","risk.events.jsonl",
        "gate.indicator.jsonl","scores.jsonl","alerts.jsonl",
        "eventblocks.jsonl","eventblocks.timeline.json","scenarios.json"
    ]
    rows=[]
    for f in files:
        p = root/f
        rows.append({
            "file": f,
            "exists": p.exists(),
            "size(B)": (p.stat().st_size if p.exists() else 0),
            "updated": (time.ctime(p.stat().st_mtime) if p.exists() else "")
        })
    st.subheader("산출물 상태 요약")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

    # (옵션) 별도 ledger.jsonl이 있다면 아래에서 표시
    ledger_rows = _read_jsonl(root/"ledger.jsonl")
    if ledger_rows:
        st.subheader("Ledger entries")
        st.dataframe(pd.DataFrame(ledger_rows).tail(200), use_container_width=True, height=320)
    else:
        st.caption("ledger.jsonl 없음(선택 파일)")
