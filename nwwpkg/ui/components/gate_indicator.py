# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import streamlit as st

# ---------- low-level loaders ----------
def _read_jsonl(p: Path):
    rows=[]
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s:
                    rows.append(json.loads(s))
    return rows

def _read_yaml(p: Path):
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        from ruamel.yaml import YAML
        y = YAML()
        return y.load(txt) or {}
    except Exception:
        import yaml
        return yaml.safe_load(txt) or {}

# ---------- frames coercion ----------
def _coerce_frames(fr) -> List[str]:
    labs=[]
    if isinstance(fr, list):
        for x in fr:
            if isinstance(x, str): labs.append(x.strip().lower())
            elif isinstance(x, dict):
                lab = (x.get("label") or x.get("frame") or x.get("name"))
                if lab: labs.append(str(lab).strip().lower())
    elif isinstance(fr, dict):
        lab = (fr.get("label") or fr.get("frame") or fr.get("name"))
        if lab: labs.append(str(lab).strip().lower())
    elif isinstance(fr, str):
        labs.append(fr.strip().lower())
    return labs

# ---------- indicator scoring ----------
def _noisy_or(vals: List[float]) -> float:
    p=1.0
    for v in vals:
        v = max(0.0, min(1.0, float(v)))
        p *= (1.0 - v)
    return 1.0 - p

def _score_indicator(fr_list, weights: Dict[str,float], alias: Dict[str,str], combine: str="linear") -> float:
    labs = []
    for lb in _coerce_frames(fr_list):
        labs.append(alias.get(lb, lb))
    if not labs:
        return 0.01  # epsilon

    uniq = list(dict.fromkeys(labs))  # 중복 제거(순서 유지)
    if combine == "noisy_or":
        vals = [max(0.0, min(1.0, float(weights.get(lb, 0.0)))) for lb in uniq]
        if not any(vals): return 0.01
        return round(_noisy_or(vals), 3)
    else:
        tot=0.0; wsum=0.0
        for lb in uniq:
            w = float(weights.get(lb, 0.0))
            if w != 0.0:
                tot += w; wsum += abs(w)
        if wsum == 0.0: return 0.01
        # [-1..1] → [0..1]로 리스케일
        sc = (tot/wsum + 1.0)/2.0
        return round(max(0.0, min(1.0, sc)), 3)

def _pick_text(r: dict) -> str:
    return (r.get("normalized") or r.get("content") or r.get("summary") or r.get("title") or "").strip()

# ---------- compute & persist ----------
def compute_gate_indicator(root: Path, cfg_path: Path) -> pd.DataFrame:
    clean   = _read_jsonl(root/"clean.jsonl")
    framesR = {r.get("id"): r for r in _read_jsonl(root/"frames.jsonl")}
    # LLM 파일명 폴백(frames_llm.jsonl ↔ frames.llm.jsonl)
    llm_path = root/"frames_llm.jsonl"
    if not llm_path.exists() and (root/"frames.llm.jsonl").exists():
        llm_path = root/"frames.llm.jsonl"
    framesL = {r.get("id"): r for r in _read_jsonl(llm_path)}

    cfg = _read_yaml(cfg_path)
    weights = cfg.get("weights", {
        "military": 0.70, "diplomacy": 0.30, "economy": 0.20, "uncertain": -0.10
    })
    alias   = cfg.get("alias", {
        "military_mobilization":"military", "troop_movement":"military",
        "mil":"military", "diplo":"diplomacy", "econ":"economy"
    })
    combine = str(cfg.get("combine", "linear")).lower()  # "linear" | "noisy_or"

    rows=[]
    for r in clean:
        i = r.get("id")
        fr = (framesR.get(i, {}) or {}).get("frames")
        if not fr:  # 규칙 프레임 없으면 LLM 프레임 사용
            fr = (framesL.get(i, {}) or {}).get("frames")

        score = _score_indicator(fr, weights, alias, combine)
        rows.append({
            "id": i,
            "normalized": _pick_text(r)[:140],
            "frames": fr,
            "indicator_score": score
        })

    df = pd.DataFrame(rows)
    # 저장(다른 스테이지에서 재사용)
    outp = root/"gate.indicator.jsonl"
    outp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    return df

# ---------- streamlit render ----------
def render_gate_indicator(root: Path, cfg_path: Path=Path("config/indicators.yaml")):
    st.header("📖 Gate – 프레임 기반 체크(Indicator)")
    st.caption("프레임(frame)→인디케이터(indicator) 매핑으로 문서별 지표 점수를 계산합니다.")

    # 재계산 버튼 + 파일mtime 비교
    btn = st.button("Recompute indicator")
    outp = root/"gate.indicator.jsonl"
    def _mt(p: Path): return os.path.getmtime(p) if p.exists() else 0
    need = btn or (_mt(outp) < max(_mt(root/"clean.jsonl"), _mt(root/"frames.jsonl"), _mt(cfg_path)))

    if need:
        df = compute_gate_indicator(root, cfg_path)
    else:
        # 캐시 로드
        df = pd.DataFrame(_read_jsonl(outp))

    if not df.empty:
        # 보기 좋게
        def _pretty(fr):
            labs = [x.get("label") or x.get("frame") or x.get("name") if isinstance(x, dict) else str(x)
                    for x in (fr or [])]
            labs = [str(x) for x in labs if x]
            return ", ".join(labs[:5])

        show = df.copy()
        show["frames_pretty"] = show["frames"].apply(_pretty)
        st.subheader("프레임 & 지표 점수")
        st.dataframe(show[["normalized","frames_pretty","indicator_score"]], use_container_width=True, height=360)
        st.success("✅ Gate 결과 저장 완료  ·  " + time.ctime(_mt(outp)))
        st.caption(f"source: {outp}")
    else:
        st.info("데이터가 비어 있습니다. clean/frames 파일을 확인하세요.")
