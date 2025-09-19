# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, time, re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st

# ------------------- IO helpers -------------------
def _read_jsonl(p: Path) -> List[dict]:
    rows=[]
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s: rows.append(json.loads(s))
    return rows

def _write_jsonl(p: Path, rows: List[dict]) -> None:
    # JSONL: 비어있으면 빈 문자열 저장
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

# ------------------- text utils -------------------
_ws_re = re.compile(r"\s+")
_tok_re = re.compile(r"[A-Za-z0-9가-힣]+")

def _norm_txt(x: str) -> str:
    if not x: return ""
    x = str(x)
    x = _ws_re.sub(" ", x).strip().lower()
    return x

def _tokens(x: str) -> List[str]:
    return _tok_re.findall(_norm_txt(x))

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / max(1.0, len(sa | sb))

def _cosine_tfidf(titles: List[str]) -> Tuple[List[List[float]] | None, List[str]]:
    """
    선택적 TF-IDF(있으면 사용). scikit-learn 미설치면 None 반환하여 Jaccard로 폴백.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
        m = vec.fit_transform([_norm_txt(t) for t in titles])
        sim = cosine_similarity(m)
        return sim.tolist(), vec.get_feature_names_out().tolist()
    except Exception:
        return None, []

def _parse_dt(s: str):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(dt): return pd.NaT
    try: return dt.tz_convert(None)
    except Exception: return dt

# ------------------- core clustering -------------------
def _cluster_alerts(df: pd.DataFrame, sim_thr: float, win_days: int, use_tfidf: bool) -> List[List[int]]:
    """
    간단한 그래프 클러스터링:
    - 두 알림의 유사도 >= sim_thr
    - 그리고 시간 차이가 win_days 일 이하이면 엣지 생성
    - 연결요소(connected components)를 블록으로 간주
    """
    n = len(df)
    if n == 0: return []

    # 전처리
    toks = [ _tokens(t) for t in df["title"].astype(str).tolist() ]
    dts  = df["dt"].tolist()

    sim_matrix = None
    if use_tfidf:
        sim_matrix, _ = _cosine_tfidf(df["title"].astype(str).tolist())

    # 인접 리스트
    adj = [[] for _ in range(n)]
    win = pd.Timedelta(days=win_days)

    for i in range(n):
        for j in range(i+1, n):
            # 시간창 체크
            ti, tj = dts[i], dts[j]
            if pd.isna(ti) or pd.isna(tj):
                time_ok = True  # 날짜 없으면 시간 제약 패스
            else:
                time_ok = abs(ti - tj) <= win
            if not time_ok:
                continue

            # 유사도
            if sim_matrix is not None:
                sim = float(sim_matrix[i][j])
            else:
                sim = _jaccard(toks[i], toks[j])

            if sim >= sim_thr:
                adj[i].append(j); adj[j].append(i)

    # 연결요소 탐색
    vis = [False]*n
    blocks=[]
    for i in range(n):
        if vis[i]: continue
        stack=[i]; comp=[]
        vis[i]=True
        while stack:
            u=stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not vis[v]:
                    vis[v]=True; stack.append(v)
        blocks.append(sorted(comp))
    return [b for b in blocks if len(b)>=1]

def _block_profile(df: pd.DataFrame, idxs: List[int]) -> dict:
    sub = df.iloc[idxs]
    # 시간
    start = sub["dt"].min()
    end   = sub["dt"].max()
    # 레벨(없으면 info)
    level_rank = {"info":0, "warning":1, "alert":2}
    if "alert_level" in sub.columns:
        lvl_series = sub["alert_level"].map(level_rank).fillna(0)
        top_level = int(lvl_series.max()) if not lvl_series.empty else 0
    else:
        top_level = 0
    inv = {v:k for k,v in level_rank.items()}
    top_level = inv.get(top_level, "info")
    # 대표 타이틀/키워드
    titles = sub["title"].astype(str).tolist() if "title" in sub.columns else []
    all_toks = []
    for t in titles:
        all_toks.extend(_tokens(t))
    freq = pd.Series(all_toks).value_counts() if all_toks else pd.Series(dtype=int)
    top_kw = freq.head(5).index.tolist()
    rep_title = max(titles, key=lambda x: len(x))[:140] if titles else ""
    return {
        "start": str(start) if pd.notna(start) else "",
        "end":   str(end)   if pd.notna(end)   else "",
        "count": int(len(sub)),
        "top_level": top_level,
        "top_keywords": top_kw,
        "rep_title": rep_title
    }

def compute_eventblocks(root: Path, sim_thr: float=0.35, win_days: int=5, use_tfidf: bool=False, min_block: int=2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    alerts.jsonl -> eventblocks.jsonl, eventblocks.timeline.json
    """
    alerts = _read_jsonl(root/"alerts.jsonl")
    if not alerts:
        # 결과 파일도 빈 상태로 생성
        _write_jsonl(root/"eventblocks.jsonl", [])
        (root/"eventblocks.timeline.json").write_text(json.dumps({"blocks":[],"members":[]}, ensure_ascii=False, indent=2), encoding="utf-8")
        return pd.DataFrame(columns=["block_id","start","end","count","top_level","top_keywords","rep_title"]), pd.DataFrame(columns=["block_id","id","title","dt","alert_level","score"])

    # 기본 DF
    df = pd.DataFrame(alerts)
    if "dt" not in df.columns:
        df["dt"] = ""
    df["dt"] = df["dt"].apply(_parse_dt)
    # 안정용: dt 없으면 일련번호 날짜 부여
    if df["dt"].isna().all():
        df["dt"] = pd.to_datetime(range(len(df)), unit="D", origin="2024-01-01")
    df = df.sort_values("dt").reset_index(drop=True)

    # 클러스터
    blocks = _cluster_alerts(df, sim_thr=sim_thr, win_days=win_days, use_tfidf=use_tfidf)
    # 최소 크기 필터
    blocks = [b for b in blocks if len(b) >= int(min_block)]

    # block_id 매핑
    blk_rows=[]
    mem_rows=[]
    for k, idxs in enumerate(blocks, start=1):
        prof = _block_profile(df, idxs)
        blk = {
            "block_id": f"blk_{k:04d}",
            **prof
        }
        blk_rows.append(blk)
        for i in idxs:
            r = df.iloc[i].to_dict()
            mem_rows.append({
                "block_id": blk["block_id"],
                "id": r.get("id"),
                "title": r.get("title",""),
                "dt": str(df.iloc[i]["dt"]),
                "alert_level": r.get("alert_level",""),
                "score": r.get("score", 0.0)
            })

    # --- 스키마 고정 + 안전 정렬 ---
    blk_cols = ["block_id","start","end","count","top_level","top_keywords","rep_title"]
    df_blk = pd.DataFrame(blk_rows, columns=blk_cols)
    if not df_blk.empty:
        # 문자열로 들어온 start/end를 datetime으로 변환해 정렬 안정화
        df_blk["_start_dt"] = pd.to_datetime(df_blk["start"], errors="coerce")
        df_blk["_end_dt"]   = pd.to_datetime(df_blk["end"], errors="coerce")
        df_blk = df_blk.sort_values(["_start_dt","_end_dt","count"], ascending=[True, True, False])
        df_blk = df_blk.drop(columns=["_start_dt","_end_dt"])
    df_mem = pd.DataFrame(mem_rows, columns=["block_id","id","title","dt","alert_level","score"])
    if not df_mem.empty:
        df_mem["dt"] = pd.to_datetime(df_mem["dt"], errors="coerce")
        df_mem = df_mem.sort_values(["block_id","dt"])

    # 저장
    _write_jsonl(root/"eventblocks.jsonl", df_blk.to_dict(orient="records"))
    (root/"eventblocks.timeline.json").write_text(
        json.dumps({"blocks": df_blk.to_dict(orient="records"),
                    "members": df_mem.to_dict(orient="records")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return df_blk, df_mem

# ------------------- streamlit render -------------------
def render_eventblocks(root: Path):
    st.header("🧩 EventBlocks — 클러스터링 & 타임라인")

    # 파라미터
    colA, colB, colC, colD = st.columns(4)
    with colA:
        sim_thr = st.slider("유사도 임계치(sim_thr)", 0.05, 0.95, 0.35, 0.05)
    with colB:
        win_days = st.slider("시간창(day)", 1, 30, 5, 1)
    with colC:
        min_block = st.number_input("블록 최소크기", min_value=1, max_value=50, value=2, step=1)
    with colD:
        use_tfidf = st.checkbox("TF-IDF 사용(있으면)", value=False)

    # 재계산 버튼
    btn = st.button("Recompute EventBlocks")
    blk_p = root/"eventblocks.jsonl"
    tim_p = root/"eventblocks.timeline.json"

    def _mt(p: Path): return os.path.getmtime(p) if p.exists() else 0
    # 결과가 없거나 alerts가 더 최신이거나 버튼 클릭 시 재계산
    need = (btn
            or (not blk_p.exists())
            or (_mt(blk_p) < _mt(root/"alerts.jsonl"))
            or (not tim_p.exists()))

    if need:
        df_blk, df_mem = compute_eventblocks(root, sim_thr=sim_thr, win_days=win_days, use_tfidf=use_tfidf, min_block=int(min_block))
    else:
        df_blk = pd.DataFrame(_read_jsonl(blk_p))
        timeline = json.loads(tim_p.read_text(encoding="utf-8", errors="ignore")) if tim_p.exists() else {"members":[],"blocks":[]}
        df_mem = pd.DataFrame(timeline.get("members", []))

    if df_blk.empty:
        st.info("이벤트 블록이 없습니다. Alerts를 먼저 생성하고(또는 임계치/시간창을 완화) 다시 시도하세요.")
        return

    # KPI
    n_blk = len(df_blk)
    n_alerts = len(df_mem)
    c1,c2 = st.columns(2)
    c1.metric("이벤트 블록 수", n_blk)
    c2.metric("블록 내 알림 수", n_alerts)

    # 타임라인(간단 막대)
    st.subheader("타임라인(개요)")
    try:
        tshow = df_blk.copy()
        tshow["start"] = pd.to_datetime(tshow["start"], errors="coerce")
        tshow["end"]   = pd.to_datetime(tshow["end"], errors="coerce")
        tshow["span_days"] = (tshow["end"] - tshow["start"]).dt.days.fillna(0).astype(int)
        tshow = tshow.sort_values("start")
        st.bar_chart(tshow.set_index("block_id")["span_days"], height=200)
    except Exception as e:
        st.caption(f"(타임라인 요약 렌더 오류: {e})")

    # 블록 표
    st.subheader("블록 요약")
    cols_show = ["block_id","start","end","count","top_level","top_keywords","rep_title"]
    for c in cols_show:
        if c not in df_blk.columns:
            df_blk[c] = "" if c != "count" else 0
    st.dataframe(df_blk[cols_show], use_container_width=True, height=300)

    # 특정 블록 선택 → 멤버 보기
    st.subheader("블록 상세")
    pick = st.selectbox("Block 선택", options=df_blk["block_id"].tolist())
    sub = df_mem[df_mem["block_id"]==pick].copy()
    if not sub.empty:
        sub["dt"] = pd.to_datetime(sub["dt"], errors="coerce")
        sub = sub.sort_values("dt")
    st.dataframe(sub[["dt","alert_level","score","title","id"]] if not sub.empty else sub,
                 use_container_width=True, height=320)

    # 다운로드
    st.download_button(
        "eventblocks.jsonl 다운로드",
        data="\n".join(json.dumps(r, ensure_ascii=False) for r in df_blk.to_dict(orient="records")),
        file_name="eventblocks.jsonl",
        mime="application/json"
    )
    st.caption(f"source: {blk_p} · updated: {time.ctime(_mt(blk_p))}")
