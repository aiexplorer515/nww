# tools/merge_frames_rule_llm.py
# -*- coding: utf-8 -*-
import sys, orjson as oj
from collections import defaultdict
from pathlib import Path
import os, tempfile, shutil

fin, fin_llm, fout = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fout.parent.mkdir(parents=True, exist_ok=True)

# 입력과 출력이 같으면 임시 파일로 쓰고 마지막에 교체
targets = {fin.resolve(), fin_llm.resolve()}
if fout.resolve() in targets:
    fd, tmpname = tempfile.mkstemp(prefix="frames_merged_", suffix=".jsonl", dir=str(fout.parent))
    os.close(fd)
    real_out = Path(tmpname)
else:
    real_out = fout

# ... (merge 로직으로 real_out에 기록)

# 기록 완료 후 교체
if real_out != fout:
    shutil.move(str(real_out), str(fout))

def readj(p):
    try:
        for l in open(p,"r",encoding="utf-8"): 
            s=l.strip()
            if s: yield oj.loads(s)
    except FileNotFoundError:
        return

def to_map(rows,key="id"):
    return {r[key]:r for r in rows}

# ==== add: 공통 헬퍼 ====
def _to_label(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("label", "name", "frame", "tag", "id"):
            v = x.get(k)
            if isinstance(v, (str, int, float)):
                return str(v).strip()
    # 기타 타입 방어
    return (str(x or "")).strip()

def _as_labels(fr_list):
    # 어떤 입력이 와도 list[str] 로 강제 변환 + 중복 제거(순서 유지)
    if isinstance(fr_list, (str, dict)):
        fr_list = [fr_list]
    elif not isinstance(fr_list, list):
        fr_list = [fr_list]
    out, seen = [], set()
    for x in fr_list:
        s = _to_label(x)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def merge(fr_rule, fr_llm):
    rule = _as_labels(fr_rule)
    llm  = _as_labels(fr_llm)

    # 간단·안전한 정책: 규칙 우선(rule-first) + LLM 보강
    out = list(rule)
    for s in llm:
        if s not in out:
            out.append(s)
    return out

if __name__=="__main__":
    fin_rule, fin_llm, fout = sys.argv[1], sys.argv[2], sys.argv[3]
    R=to_map(readj(fin_rule)); L=to_map(readj(fin_llm))
    out=[]
    for i,r in R.items():
        fr_rule=r.get("frames",[])
        fr_llm =(L.get(i,{}).get("frames",[]))
        r["frames"]=merge(fr_rule, fr_llm)
        out.append(r)
    # LLM에만 있고 룰에 없는 id 도 포함
    for i,l in L.items():
        if i in R: continue
        out.append({"id":i,"frames":merge([], l.get("frames",[]))})
    with open(fout,"w",encoding="utf-8") as w:
        for r in out: w.write(oj.dumps(r).decode()+"\n")
    print("merged ->", fout, "n=", len(out))
