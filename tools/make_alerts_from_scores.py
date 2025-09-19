# -*- coding: utf-8 -*-
"""
scores.jsonl -> alerts.jsonl 변환기
- 기본 점수: fused_score (없으면 score/conf/prob -> indicator_score -> dbn_score 순 폴백)
- 임계치(threshold)로 alert_level 결정: alert / warning / info
- 같은 id 중복 시 최고 점수만 남김
- 옵션: alerts.yaml로 임계치 설정, CLI 인자/환경변수로도 설정 가능
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from collections import defaultdict

def read_jsonl(p: Path):
    rows=[]
    if p.exists():
        with p.open(encoding="utf-8", errors="ignore") as f:
            for L in f:
                s=L.strip()
                if s: rows.append(json.loads(s))
    return rows

def write_jsonl(p: Path, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

def read_yaml(p: Path):
    if not p.exists(): return {}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        from ruamel.yaml import YAML
        y=YAML(); return y.load(txt) or {}
    except Exception:
        import yaml
        return yaml.safe_load(txt) or {}

def choose_score(r: dict) -> float:
    for k in ("fused_score","score","conf","prob","indicator_score","dbn_score"):
        if k in r:
            try:
                v=float(r[k]); 
                if v==v:  # not NaN
                    return max(0.0, min(1.0, v))
            except Exception:
                pass
    return 0.0

def pick_dt(r: dict) -> str:
    for k in ("dt","published_at","date","ts"):
        v = r.get(k)
        if v: return str(v)
    return ""

def level_by_thr(x: float, warn: float, alert: float):
    if x >= alert: return "alert", "High"
    if x >= warn:  return "warning", "Medium"
    return "info", "Low"

def main():
    parser = argparse.ArgumentParser(description="scores → alerts 변환")
    parser.add_argument("--inp", "-i", default=None, help="입력 scores.jsonl")
    parser.add_argument("--out", "-o", default=None, help="출력 alerts.jsonl")
    parser.add_argument("--alert", type=float, default=None, help="alert 임계치 (기본 0.70)")
    parser.add_argument("--warn",  type=float, default=None, help="warning 임계치 (기본 0.55)")
    parser.add_argument("--keep_info", action="store_true", help="info 레벨도 파일에 포함")
    parser.add_argument("--cfg", default="config/alerts.yaml", help="임계치 설정 YAML (옵션)")
    args = parser.parse_args()

    # 기본 경로: NWW_DATA_HOME/NWW_BUNDLE
    root = Path(os.getenv("NWW_DATA_HOME","data"))/os.getenv("NWW_BUNDLE","b01")
    inp  = Path(args.inp or (root/"scores.jsonl"))
    outp = Path(args.out or (root/"alerts.jsonl"))

    # thresholds: cfg > CLI > env > default
    cfg = read_yaml(Path(args.cfg))
    alert_thr = (
        args.alert or 
        float(os.getenv("ALERT_THR","nan") if os.getenv("ALERT_THR") else "nan") or
        float(cfg.get("thresholds",{}).get("alert", 0.70))
    )
    warn_thr  = (
        args.warn or 
        float(os.getenv("WARN_THR","nan") if os.getenv("WARN_THR") else "nan") or
        float(cfg.get("thresholds",{}).get("warn",  0.55))
    )

    rows = read_jsonl(inp)
    if not rows:
        print(f"[warn] input empty: {inp}")
        write_jsonl(outp, [])
        return

    # 같은 id 중복 시 최고점 유지
    best = {}
    for r in rows:
        rid = r.get("id")
        sc  = choose_score(r)
        if not rid: 
            continue
        if rid not in best or sc > best[rid][0]:
            best[rid] = (sc, r)

    alerts=[]
    for rid,(sc, r) in best.items():
        lvl, sev = level_by_thr(sc, warn_thr, alert_thr)
        if lvl=="info" and not args.keep_info:
            continue
        alerts.append({
            "id": rid,
            "alert_level": lvl,            # "alert" | "warning" | "info"
            "score": round(sc,3),
            "severity": sev,               # "High" | "Medium" | "Low"
            "title": r.get("title",""),
            "dt":    pick_dt(r),
            "source": "scores",            # 생성 출처
        })

    # 파일 쓰기
    write_jsonl(outp, alerts)

    # 분포 출력 및 요약 저장
    dist = defaultdict(int)
    for a in alerts: dist[a["alert_level"]] += 1
    print("alerts ->", outp, "n=", len(alerts), "dist=", dict(dist))

    (outp.with_suffix(".summary.json")
         ).write_text(json.dumps({"n": len(alerts), "dist": dict(dist)}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
