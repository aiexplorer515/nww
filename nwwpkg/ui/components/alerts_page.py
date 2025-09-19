# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import streamlit as st

# ---------- low-level IO ----------
def _read_jsonl(p: Path) -> List[dict]:
    rows=[]
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s: rows.append(json.loads(s))
    return rows

def _write_jsonl(p: Path, rows: List[dict]) -> None:
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

def _read_yaml(p: Path) -> dict:
    if not p.exists(): return {}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        from ruamel.yaml import YAML
        y=YAML(); return y.load(txt) or {}
    except Exception:
        import yaml
        return yaml.safe_load(txt) or {}

# ---------- helpers ----------
def _choose_score(r: dict) -> float:
    for k in ("fused_score","score","conf","prob","indicator_score","dbn_score"):
        if k in r:
            try:
                v=float(r[k])
                if v==v:
                    return max(0.0, min(1.0, v))
            except Exception:
                pass
    return 0.0

def _pick_dt(r: dict) -> str:
    for k in ("dt","published_at","date","ts"):
        if r.get(k): return str(r[k])
    return ""

def _level_by_thr(x: float, warn: float, alert: float):
    if x >= alert: return "alert", "High"
    if x >= warn:  return "warning", "Medium"
    return "info", "Low"

# ---------- core: scores -> alerts ----------
def compute_alerts_from_scores(root: Path, warn: float, alert: float, keep_info: bool=False) -> pd.DataFrame:
    scores = _read_jsonl(root/"scores.jsonl")
    if not scores:
        return pd.DataFrame()

    # 같은 id는 최고점만 유지
    best={}
    for r in scores:
        rid=r.get("id")
        if not rid: 
            continue
        sc=_choose_score(r)
        if rid not in best or sc>best[rid][0]:
            best[rid]=(sc,r)

    out=[]
    for rid,(sc,r) in best.items():
        lvl, sev = _level_by_thr(sc, warn, alert)
        if lvl=="info" and not keep_info:
            continue
        out.append({
            "id": rid,
            "alert_level": lvl,  # alert | warning | info
            "score": round(sc,3),
            "severity": sev,     # High | Medium | Low
            "title": r.get("title",""),
            "dt": _pick_dt(r),
            "source": "scores"
        })

    _write_jsonl(root/"alerts.jsonl", out)
    return pd.DataFrame(out)

# ---------- streamlit component ----------
def render_alerts(root: Path, cfg_path: Path=Path("config/alerts.yaml")):
    st.header("🚨 Alerts 페이지")

    # --- thresholds(임계치) UI ---
    cfg = _read_yaml(cfg_path)
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        warn = st.number_input("경보 임계치 – Warning (warn)", min_value=0.0, max_value=1.0,
                               value=float(cfg.get("thresholds",{}).get("warn", 0.55)), step=0.01, format="%.2f")
    with colB:
        alert = st.number_input("경보 임계치 – Alert (alert)", min_value=0.0, max_value=1.0,
                                value=float(cfg.get("thresholds",{}).get("alert",0.70)), step=0.01, format="%.2f")
    with colC:
        keep_info = st.checkbox("info 레벨 포함", value=False)

    btn = st.button("Recompute alerts (scores → alerts)")
    alerts_path = root/"alerts.jsonl"
    scores_path = root/"scores.jsonl"

    def _mt(p: Path): return os.path.getmtime(p) if p.exists() else 0
    need = btn or (_mt(alerts_path) < _mt(scores_path)) or (_mt(alerts_path) < _mt(cfg_path))

    if need:
        df = compute_alerts_from_scores(root, warn, alert, keep_info)
    else:
        df = pd.DataFrame(_read_jsonl(alerts_path))

    if df.empty:
        st.info("데이터가 없습니다. 먼저 Scoring에서 `scores.jsonl`을 생성하세요.")
        return

    # ----- Filters -----
    st.subheader("필터")
    fcol1, fcol2 = st.columns([2,1])
    with fcol1:
        q = st.text_input("제목 검색", "")
    with fcol2:
        lv = st.multiselect("레벨", options=["alert","warning","info"], default=["alert","warning"])
    base = df.copy()
    if q:
        base = base[base["title"].astype(str).str.contains(q, case=False, na=False)]
    if lv:
        base = base[base["alert_level"].isin(lv)]

    # ----- KPIs -----
    total = len(df)
    n_alert = int((df["alert_level"]=="alert").sum())
    n_warn  = int((df["alert_level"]=="warning").sum())
    k1,k2,k3 = st.columns(3)
    k1.metric("전체 Alerts", total)
    k2.metric("Alert(High)", n_alert)
    k3.metric("Warning(Medium)", n_warn)

    # ----- Dist/Timeseries -----
    st.subheader("분포 및 시계열")
    # 분포(막대)
    dist = base["alert_level"].value_counts().rename_axis("level").reset_index(name="count")
    if not dist.empty:
        st.bar_chart(dist, x="level", y="count", height=220)
    # 시계열(일 단위 집계)
    tmp = base.copy()
    tmp["dt"] = pd.to_datetime(tmp["dt"], errors="coerce")
    if tmp["dt"].notna().any():
        ts = (tmp.dropna(subset=["dt"])
                 .assign(date=lambda x: x["dt"].dt.date)
                 .groupby(["date","alert_level"]).size()
                 .reset_index(name="count")
                 .sort_values("date"))
        st.line_chart(ts, x="date", y="count", color="alert_level", height=260)
    else:
        st.caption("시계열: dt가 없어 그래프를 생략합니다.")

    # ----- Table & Download -----
    st.subheader("Alerts 목록")
    table = base.sort_values(["alert_level","score","dt"], ascending=[True,False,False])[
        ["dt","alert_level","score","title","id"]
    ].reset_index(drop=True)
    st.dataframe(table, use_container_width=True, height=360)

    # 다운로드
    st.download_button(
        "alerts.jsonl 다운로드",
        data="\n".join(json.dumps(r, ensure_ascii=False) for r in base.to_dict(orient="records")),
        file_name="alerts.jsonl",
        mime="application/json"
    )

    st.caption(f"source: {alerts_path} · updated: {time.ctime(_mt(alerts_path))}")
