# nwwpkg/ui/components/frames_eval.py
from __future__ import annotations
import json, os, time
from pathlib import Path
import streamlit as st
import pandas as pd

# ---------- low-level loaders ----------
def _read_jsonl(p: Path):
    rows=[]
    if not p.exists(): return rows
    with p.open(encoding="utf-8") as f:
        for L in f:
            s=L.strip()
            if s: rows.append(json.loads(s))
    return rows

def _to_label(x):
    if isinstance(x, str): return x.strip()
    if isinstance(x, dict):
        for k in ("label","frame","name","tag","id"):
            v=x.get(k)
            if isinstance(v,(str,int,float)): return str(v).strip()
    return ""

def _as_list(fr):
    if fr is None: return []
    if isinstance(fr,list): return fr
    return [fr]

def _coerce(fr_list):
    out=[]
    for x in _as_list(fr_list):
        if isinstance(x, dict):
            lab=_to_label(x)
            try: conf=float(x.get("conf", x.get("score",0.0)))
            except: conf=0.0
            if lab: out.append({"label":lab,"conf":conf})
        else:
            lab=_to_label(x)
            if lab: out.append({"label":lab,"conf":0.0})
    return out

def _top1(fr_list):
    fr=_coerce(fr_list)
    if not fr: return (None,0.0)
    best=max(fr, key=lambda d: float(d.get("conf",0.0)))
    return (best["label"], float(best.get("conf",0.0)))

def _index_by_id(rows): return {r.get("id"): r for r in rows if r.get("id")}

def _coverage(rows):
    n=len(rows)
    if n==0: return 0.0
    hit=sum(1 for r in rows if _top1(r.get("frames"))[0])
    return round(hit/max(1,n),3)

# --- BOM/다중 인코딩 허용 리더 ---
def _read_json_any_encoding(p: Path) -> dict:
    b = p.read_bytes()
    if not b: return {}
    if b.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        return json.loads(b[3:].decode("utf-8"))
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):  # UTF-16
        return json.loads(b.decode("utf-16"))
    for enc in ("utf-8","utf-8-sig"):
        try: return json.loads(b.decode(enc))
        except Exception: pass
    return json.loads(b.decode("latin-1"))

# ---------- core evaluator ----------
def compute_eval(fr_rule: Path, fr_llm: Path, fr_merge: Path) -> dict:
    R=_read_jsonl(fr_rule); L=_read_jsonl(fr_llm); M=_read_jsonl(fr_merge)
    idR=_index_by_id(R); idL=_index_by_id(L); idM=_index_by_id(M)
    ids=sorted(set(idR)&set(idL))

    agree_rl=agree_rm=agree_lm=0; n_agree=0
    conf_rule=[]; conf_llm=[]; conf_merge=[]

    for i in ids:
        r, l, m = idR[i], idL[i], idM.get(i,{})
        lr,cr=_top1(r.get("frames")); conf_rule.append(cr)
        ll,cl=_top1(l.get("frames")); conf_llm.append(cl)
        lm,cm=_top1(m.get("frames")) if m else (None,0.0); conf_merge.append(cm)

        if lr and ll:
            n_agree+=1
            if lr==ll: agree_rl+=1
        if lr and lm:
            if lr==lm: agree_rm+=1
        if ll and lm:
            if ll==lm: agree_lm+=1

    return {
        "counts":{"rule":len(R),"llm":len(L),"merged":len(M),
                  "ids_intersection_rl":len(ids),"agree_pairs_count":n_agree},
        "coverage":{"rule":_coverage(R),"llm":_coverage(L),"merged":_coverage(M)},
        "avg_conf":{"rule":round(sum(conf_rule)/max(1,len(conf_rule)),3),
                    "llm": round(sum(conf_llm)/max(1,len(conf_llm)),3),
                    "merged":round(sum(conf_merge)/max(1,len(conf_merge)),3)},
        "agreement":{"rule_vs_llm":round(agree_rl/max(1,n_agree),3),
                     "rule_vs_merged":round(agree_rm/max(1,len(ids)),3),
                     "llm_vs_merged": round(agree_lm/max(1,len(ids)),3)}
    }

# ---------- streamlit component ----------
def render_frames_eval(root: Path):
    rule = root/"frames.rule.jsonl"
    llm  = root/"frames_llm.jsonl"
    if not llm.exists() and (root/"frames.llm.jsonl").exists():
        llm = root/"frames.llm.jsonl"
    merged = root/"frames.jsonl"
    eval_path = root/"frames.eval.json"

    st.subheader("Frames Evaluation (rule vs LLM vs merged)")
    refresh = st.button("Recompute eval")
    _mt=lambda p: os.path.getmtime(p) if p.exists() else 0
    need = refresh or (not eval_path.exists()) or (max(_mt(rule),_mt(llm),_mt(merged))>_mt(eval_path))

    if need:
        result = compute_eval(rule, llm, merged)
        eval_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        try:
            result = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = _read_json_any_encoding(eval_path)

    cov, conf, agr = result["coverage"], result["avg_conf"], result["agreement"]
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Coverage rule/llm/merged", f"{cov['rule']:.0%} / {cov['llm']:.0%} / {cov['merged']:.0%}")
    with c2: st.metric("Avg conf rule/llm/merged", f"{conf['rule']:.2f} / {conf['llm']:.2f} / {conf['merged']:.2f}")
    with c3: st.metric("Agreement r↔l / r↔m / l↔m", f"{agr['rule_vs_llm']:.0%} / {agr['rule_vs_merged']:.0%} / {agr['llm_vs_merged']:.0%}")

    st.write("### Coverage");       st.bar_chart(pd.DataFrame(cov,  index=[0]).T.rename(columns={0:"value"}))
    st.write("### Avg confidence"); st.bar_chart(pd.DataFrame(conf, index=[0]).T.rename(columns={0:"value"}))
    st.write("### Agreement");      st.bar_chart(pd.DataFrame(agr,  index=[0]).T.rename(columns={0:"value"}))
    st.caption(f"source: {eval_path} · updated: {time.ctime(_mt(eval_path))}")
