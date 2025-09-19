# tools/risk_engine.py
# -*- coding: utf-8 -*-
import os, json, re
from pathlib import Path
from collections import defaultdict

# ---- paths ----
ROOT = Path(os.getenv("NWW_DATA_HOME", "data")) / os.getenv("NWW_BUNDLE", "b01")
CFG  = Path("config/checklists.yaml")

# ok 이벤트를 risk.events.jsonl에 포함할지 (기본: 미포함)
KEEP_OK = os.getenv("RISK_KEEP_OK", "0") == "1"

# ---- YAML loader (ruamel -> pyyaml 폴백) ----
def load_yaml(p: Path):
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        from ruamel.yaml import YAML
        y = YAML()
        return y.load(txt) or []
    except Exception:
        import yaml
        return yaml.safe_load(txt) or []

# ---- readers ----
def read_jsonl(p: Path):
    rows = []
    if not p.exists(): 
        return rows
    with p.open(encoding="utf-8", errors="ignore") as f:
        for L in f:
            s = L.strip()
            if s:
                rows.append(json.loads(s))
    return rows

# ---- helpers ----
def top1(fr_list):
    """frames: list[dict(label/conf)] or list[str] -> (label, conf)"""
    if not isinstance(fr_list, list) or not fr_list:
        return (None, 0.0)
    best, bc = None, -1.0
    for x in fr_list:
        if isinstance(x, str):
            lab, c = x, 0.0
        elif isinstance(x, dict):
            lab = x.get("label") or x.get("frame") or x.get("name")
            try:
                c = float(x.get("conf", x.get("score", 0.0)))
            except:
                c = 0.0
        else:
            continue
        if lab and c > bc:
            best, bc = str(lab).lower(), c
    return best, max(0.0, bc)

def noisy_or(scores):
    p = 1.0
    for s in scores:
        s = max(0.0, min(1.0, float(s)))
        p *= (1.0 - s)
    return 1.0 - p

def linear(map_scores, weights):
    tot, wsum = 0.0, 0.0
    for k, v in map_scores.items():
        w = float(weights.get(k, 0.0))
        tot  += w * float(v)
        wsum += w
    return tot / max(1e-9, wsum)

def doc_text(r):
    keys = ("title", "content", "text", "body", "summary", "desc")
    return " ".join(str(r.get(k, "")) for k in keys).lower()

def evaluate_signal(sig, frames_union, text):
    """신호 매칭: any_frames / any_keywords"""
    sc   = float(sig.get("score", 0.0))
    cond = sig.get("when", {}) or {}
    ok   = False

    if "any_frames" in cond:
        targets = {t.lower() for t in cond["any_frames"] or []}
        if targets & frames_union:
            ok = True
    if "any_keywords" in cond and not ok:
        for k in cond["any_keywords"] or []:
            if k and k.lower() in text:
                ok = True
                break
    return (ok, sc)

# ---- main ----
def main():
    # inputs
    clean  = read_jsonl(ROOT / "clean.jsonl")
    frames = {r.get("id"): r for r in read_jsonl(ROOT / "frames.jsonl")}

    # frames_llm.jsonl ↔ frames.llm.jsonl 폴백
    llm_path = ROOT / "frames_llm.jsonl"
    if not llm_path.exists() and (ROOT / "frames.llm.jsonl").exists():
        llm_path = ROOT / "frames.llm.jsonl"
    llm = {r.get("id"): r for r in read_jsonl(llm_path)}

    rules = load_yaml(CFG)

    events_out = []
    # ✅ ok 포함, 안전 누적
    topic_acc = defaultdict(lambda: {
        "n": 0, "ok": 0, "warn": 0, "alert": 0, "crisis": 0, "sum_score": 0.0
    })

    for doc in clean:
        did = doc.get("id")
        if not did:
            continue

        fr_rule = frames.get(did, {}).get("frames", [])
        fr_llm  = llm.get(did,    {}).get("frames", [])

        # 프레임 라벨 집합
        fu = set()
        for frs in (fr_rule, fr_llm):
            if isinstance(frs, list):
                for x in frs:
                    if isinstance(x, str):
                        fu.add(x.lower())
                    elif isinstance(x, dict):
                        lab = (x.get("label") or x.get("frame") or x.get("name"))
                        if lab:
                            fu.add(str(lab).lower())

        txt = doc_text(doc)

        for rule in rules:
            topic = rule.get("topic") or ""
            if not topic:
                continue

            scores_list = []
            scores_map  = {}   # linear용
            fired       = []

            for sig in rule.get("signals", []) or []:
                ok, sc = evaluate_signal(sig, fu, txt)
                if ok:
                    sid = sig.get("id") or f"sig_{len(scores_list)}"
                    scores_list.append(sc)
                    scores_map[sid] = sc
                    fired.append(sid)

            if not scores_list:
                continue

            comb = str(rule.get("combine", "noisy_or")).lower()
            if comb == "linear":
                sc = linear(scores_map, rule.get("linear_weights", {}))
            else:
                sc = noisy_or(scores_list)

            th     = rule.get("thresholds", {}) or {}
            warn   = float(th.get("warn",   0.40))
            alert  = float(th.get("alert",  0.70))
            crisis = float(th.get("crisis", 0.85))

            level = "ok"
            if sc >= crisis: level = "crisis"
            elif sc >= alert: level = "alert"
            elif sc >= warn:  level = "warn"

            # 이벤트 파일 기록 (ok는 기본적으로 제외)
            if KEEP_OK or level != "ok":
                events_out.append({
                    "id": did, "topic": topic, "score": round(sc, 3), "level": level,
                    "title": doc.get("title") or "",
                    "signals": fired,
                    "ts": doc.get("published_at") or doc.get("date") or ""
                })

            # ✅ 안전 누적
            acc = topic_acc[topic]
            acc["n"]         += 1
            acc["sum_score"] += sc
            if level in ("warn", "alert", "crisis"):
                acc[level] += 1
            else:
                acc["ok"]  += 1

    # outputs
    evp = ROOT / "risk.events.jsonl"
    with evp.open("w", encoding="utf-8") as w:
        for e in events_out:
            w.write(json.dumps(e, ensure_ascii=False) + "\n")

    topp   = ROOT / "risk.topics.json"
    topics = []
    for t, a in topic_acc.items():
        avg = a["sum_score"] / max(1, a["n"])
        topics.append({
            "topic": t, "n": a["n"], "avg": round(avg, 3),
            "ok": a["ok"], "warn": a["warn"], "alert": a["alert"], "crisis": a["crisis"]
        })
    topics.sort(key=lambda x: x["avg"], reverse=True)
    topp.write_text(json.dumps({"topics": topics}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
