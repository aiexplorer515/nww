# -*- coding: utf-8 -*-
"""
MVP 자동 판정기 (frames / alerts / network / schema / overview)
- 입력: data/<bundle> 아래 JSONL/JSON
- 출력: PASS/FAIL, 수치, 총점(progress score)
환경변수(옵션):
  NWW_QA_CHARS_MIN=800, NWW_QA_SENTS_MIN=8
  NWW_FRAME_COVER_MIN=0.8, NWW_FRAME_CONF_MIN=0.6
  NWW_NETWORK_N_MIN=30, NWW_NETWORK_E_MIN=60
  NWW_SOURCE_COV_MIN=0.9, NWW_REPORTER_COV_MIN=0.6
가중치(권장): frame 30, alerts 20, network 20, schema 20, overview 10
"""
import os, json, math, re, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(os.getenv("NWW_DATA_HOME","data"))/os.getenv("NWW_BUNDLE","b01")

# ---- thresholds (configurable via env) ----
def _env_float(k, d): 
    try: return float(os.getenv(k, d))
    except: return d

QA_CH_MIN   = int(float(os.getenv("NWW_QA_CHARS_MIN",   "800")))
QA_SE_MIN   = int(float(os.getenv("NWW_QA_SENTS_MIN",   "8")))
FR_COVER_MIN= _env_float("NWW_FRAME_COVER_MIN", 0.80)
FR_CONF_MIN = _env_float("NWW_FRAME_CONF_MIN",  0.60)
NW_N_MIN    = int(float(os.getenv("NWW_NETWORK_N_MIN", "30")))
NW_E_MIN    = int(float(os.getenv("NWW_NETWORK_E_MIN", "60")))
SRC_COV_MIN = _env_float("NWW_SOURCE_COV_MIN",  0.90)
REP_COV_MIN = _env_float("NWW_REPORTER_COV_MIN",0.60)

WEIGHTS = {"frame":30, "alerts":20, "network":20, "schema":20, "overview":10}

def read_jsonl(p: Path):
    rows=[]; errors=0
    if not p.exists(): return rows, errors
    with p.open(encoding="utf-8-sig") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            try:
                rows.append(json.loads(s))
            except Exception:
                errors+=1
    return rows, errors

def read_json(p: Path):
    if not p.exists(): return {}, 0
    try:
        return json.loads(p.read_text(encoding="utf-8-sig")), 0
    except Exception:
        return {}, 1

def top_label(fr_list):
    # frames: list[dict] with keys 'label'/'frame' & 'conf'
    if not isinstance(fr_list, list) or not fr_list:
        return None, 0.0
    best=None; bestc=-1.0
    for d in fr_list:
        if not isinstance(d, dict): continue
        lab = d.get("label") or d.get("frame")
        try:
            c = float(d.get("conf", d.get("score", 0.0)))
        except:
            c = 0.0
        if lab and c>bestc:
            best, bestc = lab, c
    return best, max(bestc,0.0)

def coverage(rows, key):
    n=len(rows)
    if n==0: return 0.0
    hit=0
    for r in rows:
        v=r.get(key)
        if isinstance(v, str) and v.strip():
            hit+=1
        elif isinstance(v, list) and len([x for x in v if str(x).strip()])>0:
            hit+=1
    return round(hit/n, 3)

def dedup_by_id(rows):
    seen=set(); out=[]
    for r in rows:
        i=r.get("id")
        if i and i not in seen:
            seen.add(i); out.append(r)
    return out

def pct(a,b): 
    return round(100.0*a/max(1,b),1)

# ---------- load all ----------
clean,  e_clean  = read_jsonl(ROOT/"clean.jsonl")
frames, e_frames = read_jsonl(ROOT/"frames.jsonl")
scores, e_scores = read_jsonl(ROOT/"scores.jsonl")
alerts, e_alerts = read_jsonl(ROOT/"alerts.jsonl")
kw,     e_kw     = read_jsonl(ROOT/"keywords.jsonl")
net,    e_net    = read_json(ROOT/"network.json")
frames_llm, _    = read_jsonl(ROOT/"frames_llm.jsonl")  # 있을 때만 사용

clean = dedup_by_id(clean)
frames= dedup_by_id(frames)
scores= dedup_by_id(scores)
alerts= dedup_by_id(alerts)

# ---------- 1) Frame (coverage + proxy quality) ----------
n_clean = len(clean)
id2fr   = {r.get("id"): r for r in frames}
cover_cnt= sum(1 for r in clean if r.get("id") in id2fr and top_label(id2fr[r["id"]].get("frames"))[0])
cover = round(cover_cnt/max(1,n_clean),3)

# avg top conf
top_confs=[]
for r in clean:
    fr = id2fr.get(r.get("id"), {}).get("frames")
    lab, conf = top_label(fr)
    if lab: top_confs.append(conf)
avg_conf = round(sum(top_confs)/max(1,len(top_confs)),3)

# agreement (optional) if frames_llm exists
agree = None
if frames_llm:
    id2llm = {r.get("id"): r for r in frames_llm}
    hit=0; n=0
    for i, r in id2fr.items():
        ll = id2llm.get(i, {})
        a, _ = top_label(r.get("frames"))
        b, _ = top_label(ll.get("frames"))
        if a is None or b is None: continue
        n += 1
        hit += int(str(a).strip()==str(b).strip())
    agree = round(hit/max(1,n),3)

frame_pass = (cover>=FR_COVER_MIN) and ((avg_conf>=FR_CONF_MIN) or ((agree is not None) and (agree>=0.6)))

# ---------- 2) Alerts ----------
al_dist = Counter(str(r.get("alert_level","")).lower() for r in alerts)
alerts_total = sum(al_dist.values())
alerts_pass  = alerts_total >= 1  # 최소 1개

# ---------- 3) Network ----------
nodes = len(net.get("nodes",{})) if isinstance(net, dict) else 0
edges = len(net.get("edges",[])) if isinstance(net, dict) else 0
network_pass = (nodes>=NW_N_MIN) and (edges>=NW_E_MIN)

# ---------- 4) Schema (source, reporter) ----------
# clean.enriched.jsonl 가 있으면 그걸 우선 참고
clean_enriched, _ = read_jsonl(ROOT/"clean.enriched.jsonl")
base = clean_enriched if clean_enriched else clean

def has_source(r):
    v = r.get("source") or r.get("publisher") or r.get("media")
    return isinstance(v,str) and v.strip()

def has_reporter(r):
    if isinstance(r.get("reporter"), str) and r["reporter"].strip(): return True
    reps = r.get("reporters")
    if isinstance(reps, list) and any(str(x).strip() for x in reps): return True
    return False

src_cov = round(sum(1 for r in base if has_source(r))/max(1,len(base)),3)
rep_cov = round(sum(1 for r in base if has_reporter(r))/max(1,len(base)),3)
schema_pass = (src_cov>=SRC_COV_MIN) and (rep_cov>=REP_COV_MIN)

# ---------- 5) Overview Robustness ----------
# (파서 에러가 없고, QA 지표가 계산되면 PASS)
parse_errors = e_clean+e_frames+e_scores+e_alerts+e_kw+e_net
qa_pass_cnt = sum(1 for r in clean if int(r.get("num_chars",0))>=QA_CH_MIN and int(r.get("num_sents",0))>=QA_SE_MIN)
qa_rate     = round(qa_pass_cnt/max(1,n_clean),3)
overview_pass = (parse_errors==0) and (qa_rate>=0.5)  # 최소 50%는 길이/문장수 확보

# ---------- Score ----------
score = 0
score += WEIGHTS["frame"]   if frame_pass   else 0
score += WEIGHTS["alerts"]  if alerts_pass  else 0
score += WEIGHTS["network"] if network_pass else 0
score += WEIGHTS["schema"]  if schema_pass  else 0
score += WEIGHTS["overview"]if overview_pass else 0

result = {
  "bundle": str(ROOT),
  "files": {
    "clean.jsonl": n_clean,
    "frames.jsonl": len(frames),
    "scores.jsonl": len(scores),
    "alerts.jsonl": len(alerts),
    "keywords.jsonl": len(kw),
    "network.json": {"nodes": nodes, "edges": edges}
  },
  "criteria": {
    "frame": {
      "pass": frame_pass, "coverage": cover, "avg_top_conf": avg_conf,
      **({"llm_agreement": agree} if agree is not None else {})
    },
    "alerts": {
      "pass": alerts_pass, "counts": dict(al_dist), "total": alerts_total
    },
    "network": {
      "pass": network_pass, "nodes": nodes, "edges": edges,
      "min_required": {"nodes": NW_N_MIN, "edges": NW_E_MIN}
    },
    "schema": {
      "pass": schema_pass, "source_cov": src_cov, "reporter_cov": rep_cov,
      "min_required": {"source_cov": SRC_COV_MIN, "reporter_cov": REP_COV_MIN}
    },
    "overview": {
      "pass": overview_pass, "qa_rate": qa_rate, "parse_errors": parse_errors,
      "min_required": {"qa_rate": 0.5, "parse_errors": 0}
    }
  },
  "progress_score": score,  # 0~100
  "weights": WEIGHTS
}

def _flag(b): return "PASS" if b else "FAIL"

# pretty print
print("="*66)
print("MVP AUTO-CHECK @", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("bundle:", ROOT)
print("-"*66)
print(f"[frame    ] {_flag(frame_pass)} | cover={cover:.0%}  avg_conf={avg_conf:.2f}" + (f"  llm_agree={agree:.0%}" if agree is not None else ""))
print(f"[alerts   ] {_flag(alerts_pass)} | total={alerts_total}  dist={dict(al_dist)}")
print(f"[network  ] {_flag(network_pass)} | nodes={nodes} (≥{NW_N_MIN}), edges={edges} (≥{NW_E_MIN})")
print(f"[schema   ] {_flag(schema_pass)} | source_cov={src_cov:.0%} (≥{int(SRC_COV_MIN*100)}%), reporter_cov={rep_cov:.0%} (≥{int(REP_COV_MIN*100)}%)")
print(f"[overview ] {_flag(overview_pass)} | qa_rate={qa_rate:.0%} (chars≥{QA_CH_MIN}, sents≥{QA_SE_MIN}), parse_errors={parse_errors}")
print("-"*66)
print(f"PROGRESS SCORE: {score}/100   (weights={WEIGHTS})")
print("="*66)

# JSON dump (파이프라인에서 후처리용)
print(json.dumps(result, ensure_ascii=False))
