# tools/eval_frames.py
# -*- coding: utf-8 -*-
import sys, orjson as oj
from collections import Counter

def top1(frames): 
    return (frames[0]["label"] if frames else "없음(None)")

def readj(p):
    return [oj.loads(l) for l in open(p,"r",encoding="utf-8")] if p else []

if __name__=="__main__":
    f_rule, f_llm, f_merge = sys.argv[1], sys.argv[2], sys.argv[3]
    R={r["id"]:r for r in readj(f_rule)}
    L={r["id"]:r for r in readj(f_llm)}
    M={r["id"]:r for r in readj(f_merge)}
    ids = sorted(set(R)|set(L)|set(M))
    def acc(A,B):
        hit=0; n=0
        for i in ids:
            a=top1((A.get(i) or {}).get("frames",[]))
            b=top1((B.get(i) or {}).get("frames",[]))
            if a is None or b is None: continue
            hit+= int(a==b); n+=1
        return hit, n, round(hit/max(n,1)*100,1)
    # 상호 일치율
    hr, nr, ar = acc(R,L)
    hm, nm, am = acc(R,M)
    print(f"Rule vs LLM  : {hr}/{nr}  = {ar}%")
    print(f"Rule vs Merge: {hm}/{nm}  = {am}%")
