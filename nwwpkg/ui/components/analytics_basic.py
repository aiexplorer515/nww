# nwwpkg/ui/components/analytics_basic.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, re
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------- 공통: JSONL 로더 (BOM/UTF-16 안전) ----------
def read_jsonl_any(p: Path):
    rows=[]
    if not p.exists(): return rows
    b = p.read_bytes()
    # BOM 제거/정규화
    if b.startswith(b"\xef\xbb\xbf"): b=b[3:]
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        b = b.decode("utf-16").encode("utf-8")
    for line in b.decode("utf-8", errors="ignore").splitlines():
        s=line.strip()
        if not s: continue
        try: rows.append(json.loads(s))
        except Exception: pass
    return rows

# ---------- 키워드 표준화 ----------
_HANGUL = re.compile(r"[가-힣]")

def load_keywords_df(root: Path, han_only: bool=False) -> pd.DataFrame:
    rows = read_jsonl_any(root/"keywords.jsonl")
    data=[]
    for r in rows:
        word = (r.get("keyword") or r.get("word") or r.get("term") or r.get("tag") or "").strip()
        if not word and isinstance(r.get("keywords"), list) and r["keywords"]:
            word = str(r["keywords"][0]).strip()
        if not word: continue
        if han_only and not _HANGUL.search(word):
            continue
        freq = r.get("count") or r.get("freq") or r.get("frequency") or 1
        try: freq = int(freq)
        except: 
            try: freq = float(freq)
            except: freq = 1
        data.append({"word": word.lower(), "freq": freq})
    if not data:
        return pd.DataFrame(columns=["word","freq"])
    df = pd.DataFrame(data).groupby("word", as_index=False)["freq"].sum()
    df = df.sort_values("freq", ascending=False)
    return df

# ---------- 감정/정서 표준화 ----------
SENT_MAP = {
    "pos":"pos","positive":"pos","joy":"pos","trust":"pos","optimistic":"pos",
    "neg":"neg","negative":"neg","anger":"neg","fear":"neg","sadness":"neg","disgust":"neg","pessimistic":"neg",
    "neu":"neu","neutral":"neu","mixed":"neu","other":"neu","others":"neu","unknown":"neu","none":"neu"
}

def load_sentiment_counts(root: Path) -> pd.Series:
    rows = read_jsonl_any(root/"scores.jsonl")
    cnt = {"pos":0, "neg":0, "neu":0}
    for r in rows:
        s = r.get("sentiment") or r.get("label") or r.get("polarity") or r.get("emotion")
        if isinstance(s, dict):
            s = s.get("label") or s.get("name")
        s = str(s or "").strip().lower()
        s = SENT_MAP.get(s, "neu")
        cnt[s] += 1
    return pd.Series(cnt)

# ---------- 렌더러 ----------
def render_keywords_and_sentiment(root: Path, topn: int = 20):
    st.subheader("📂 키워드 Top-N (한글)")
    col1, col2 = st.columns([1,3])
    with col1:
        han_only = st.checkbox("한글만 보기", value=True)
    df_kw = load_keywords_df(root, han_only=han_only)
    # ✅ UI에는 오직 word/freq만 노출
    st.dataframe(df_kw.loc[:, ["word","freq"]].head(topn), use_container_width=True)
    st.caption(f"소스: {root/'keywords.jsonl'}")

    st.subheader("🧪 감정 분포 (실제)")
    s = load_sentiment_counts(root).rename(index={"pos":"긍정","neg":"부정","neu":"중립"})
    st.bar_chart(s)  # pie 대신 막대 (가독성↑)
