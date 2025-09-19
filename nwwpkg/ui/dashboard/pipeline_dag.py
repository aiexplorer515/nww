import streamlit as st
from pathlib import Path
from nwwpkg.utils.io import count_lines

def render_pipeline_dag(bundle_path: Path):
    """
    📦 Pipeline DAG (처리 단계 현황)
    """
    STAGES = [
        ("ingest", "ingest.jsonl"),
        ("clean", "clean.jsonl"),
        ("dedup", ("clean.dedup.jsonl", "clean.cheap.jsonl")),
        ("keyword", "kyw.jsonl"),
        ("kboost", "kyw_boosted.jsonl"),
        ("frame", "frames.jsonl"),
        ("score", "scores.jsonl"),
        ("alert", "alerts.jsonl"),
    ]

    cols = st.columns(len(STAGES))
    for i, (name, fns) in enumerate(STAGES):
        if isinstance(fns, tuple):
            exist = [(bundle_path/x) for x in fns if (bundle_path/x).exists()]
            n = max((count_lines(p) for p in exist), default=0)
        else:
            p = bundle_path / fns
            n = count_lines(p)
        icon = "✅" if n > 0 else "❌"
        cols[i].metric(label=name, value=n, help=f"{icon} {name}")
