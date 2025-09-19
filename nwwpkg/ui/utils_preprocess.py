import json
from pathlib import Path
import streamlit as st

def load_prep_qc(bundle: str) -> dict:
    p = Path(f"data/bundles/{bundle}/prep.qc.json")
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def health_badge(qc: dict) -> list[str]:
    badges = []
    if not qc: 
        return ["⚠️ 전처리 품질 리포트 없음 (prep.qc.json)"]
    if qc.get("chars_median", 0) < 600:
        badges.append("⚠️ chars_median<600 (본문 부족)")
    if qc.get("sents_median", 0) < 6:
        badges.append("⚠️ sents_median<6 (문장 부족)")
    if qc.get("short_docs_pct", 100) > 30:
        badges.append("⚠️ short_docs_pct>30% (짧은 문서 과다)")
    return badges
