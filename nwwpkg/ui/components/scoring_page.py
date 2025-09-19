# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import streamlit as st

# ---------------- low-level loaders ----------------
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

# ---------------- helpers ----------------
def _parse_dt(r: dict):
    """published_at / date / ts 중 하나를 안전 파싱"""
    for k in ("published_at","date","ts"):
        v = r.get(k)
        if v:
            dt = pd.to_datetime(v, errors="coerce", utc=True)
            if pd.notna(dt):
                try:
                    return dt.tz_convert(None)
                except Exception:
                    return dt
    return pd.NaT

def _noisy_or(vals: List[float]) -> float:
    p=1.0
    for v in vals:
        v=max(0.0, min(1.0, float(v)))
        p *= (1.0 - v)
    return 1.0 - p

def _fuse_scores(ind_sc: float, dbn_sc: float, combine: str, w_ind: float, w_dbn: float) -> float:
    ind_sc = max(0.0, min(1.0, float(ind_sc or 0.0)))
    dbn_sc = max(0.0, min(1.0, float(dbn_sc or 0.0)))
    if combine == "noisy_or":
        return round(_noisy_or([ind_sc, dbn_sc]), 3)
    # linear (기본): 가중합
    denom = max(1e-9, abs(w_ind)+abs(w_dbn))
    sc = (w_ind*ind_sc + w_dbn*dbn_sc)/denom
    return round(max(0.0, min(1.0, sc)), 3)

def _severity(x: float, thr: dict) -> str:
    if x >= float(thr.get("high",   0.75)): return "High"
    if x >= float(thr.get("medium", 0.55)): return "Medium"
    if x >= float(thr.get("low",    0.30)): return "Low"
    return "None"

def _shift(prev: float, curr: float, delta_thr: float) -> dict:
    if prev is None: 
        return {"changed": [], "shift_detected": False, "shift_score": 0.0}
    delta = (curr - prev)
    return {
        "changed": [] if abs(delta) < delta_thr else ["fused_score"],
        "shift_detected": abs(delta) >= delta_thr,
        "shift_score": round(abs(delta), 3)
    }

# ---------------- core compute ----------------
def compute_scoring(root: Path, cfg_path: Path) -> pd.DataFrame:
    """gate.indicator + dbn.pred(옵션) -> scores.jsonl 생성"""
    clean = _read_jsonl(root/"clean.jsonl")
    gind  = {r.get("id"): r for r in _read_jsonl(root/"gate.indicator.jsonl")}
    # DBN 예측은 선택
    dbn_p = {r.get("id"): r for r in _read_jsonl(root/"dbn.pred.jsonl")}

    cfg = _read_yaml(cfg_path)
    combine = str(cfg.get("combine","linear")).lower()  # "linear" | "noisy_or"
    w_ind   = float(cfg.get("weights",{}).get("indicator", 0.6))
    w_dbn   = float(cfg.get("weights",{}).get("dbn",       0.4))
    thr     = cfg.get("severity", {"low":0.30,"medium":0.55,"high":0.75})
    dthr    = float(cfg.get("shift_delta", 0.15))

    # 타임스탬프 붙여 정렬
    base = []
    for r in clean:
        dt = _parse_dt(r)
        base.append({"id": r.get("id"), "dt": dt, "title": r.get("title","")})
    df = pd.DataFrame(base)
    if df["dt"].isna().all():
        # 날짜 없으면 문서 순서로 가짜 날짜 생성(시각화용)
        df["dt"] = pd.to_datetime(range(len(df)), unit="D", origin="2024-01-01")
    df = df.sort_values("dt").reset_index(drop=True)

    rows=[]
    prev_fused = None
    for _, row in df.iterrows():
        i   = row["id"]
        ind = float((gind.get(i, {}) or {}).get("indicator_score", 0.0))
        dbn = float((dbn_p.get(i, {}) or {}).get("dbn_score", 0.0))
        fused = _fuse_scores(ind, dbn, combine, w_ind, w_dbn)
        sev   = _severity(fused, thr)
        sh    = _shift(prev_fused, fused, dthr)
        prev_fused = fused

        rows.append({
            "id": i,
            "dt": str(row["dt"]),  # 저장은 문자열
            "indicator_score": round(ind,3),
            "dbn_score": round(dbn,3),
            "fused_score": fused,
            "severity": sev,
            "impact": {"fused_score": fused, "impact_summary": [], "severity": sev},  # 확장 포인트
            "shift": sh
        })

    # 저장
    outp = root/"scores.jsonl"
    _write_jsonl(outp, rows)
    return pd.DataFrame(rows)

# ---------------- streamlit render ----------------
def render_scoring(root: Path, cfg_path: Path=Path("config/scoring.yaml")):
    st.header("📊 Scoring – DBN 추론 + 융합 점수 + 영향/시프트")

    btn = st.button("Recompute scores")
    outp = root/"scores.jsonl"

    def _mt(p: Path): return os.path.getmtime(p) if p.exists() else 0
    need = btn or (_mt(outp) < max(_mt(root/"gate.indicator.jsonl"), _mt(cfg_path)))

    # dbn.pred.jsonl이 있을 때만 mt 비교에 포함
    dbn_p = root/"dbn.pred.jsonl"
    if dbn_p.exists():
        need = need or (_mt(outp) < _mt(dbn_p))

    if need:
        df = compute_scoring(root, cfg_path)
    else:
        df = pd.DataFrame(_read_jsonl(outp))

    if df.empty:
        st.info("데이터가 비어 있습니다. gate.indicator.jsonl / dbn.pred.jsonl을 확인하세요.")
        return

    # 요약 표 (상단 100건만)
    st.subheader("점수 & 분석 요약")
    show = df[["indicator_score","dbn_score","fused_score","impact","shift"]].copy()
    st.dataframe(show.head(100), use_container_width=True, height=280)

    # 시계열
    st.subheader("점수 시계열")
    try:
        tmp = df.copy()
        tmp["dt"] = pd.to_datetime(tmp["dt"], errors="coerce")
        ts = pd.melt(
            tmp[["dt","indicator_score","dbn_score","fused_score"]],
            id_vars=["dt"], var_name="variable", value_name="value"
        ).sort_values("dt")
        st.line_chart(ts, x="dt", y="value", color="variable", height=360)
    except Exception as e:
        st.caption(f"(시계열 렌더 오류: {e})")

    st.caption(f"source: {outp}  ·  updated: {time.ctime(_mt(outp))}")
