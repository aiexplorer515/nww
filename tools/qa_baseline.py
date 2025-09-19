# -*- coding: utf-8 -*-
import os, json
from pathlib import Path

B   = os.getenv("NWW_BUNDLE", "b01")
ROOT= Path(os.getenv("NWW_DATA_HOME", "data"))/B
p   = ROOT/"clean.jsonl"

# 데모 임계(환경변수)와 베이스라인(고정) 동시 계산
CH_DEMO = int(os.getenv("NWW_QA_CHARS_MIN", "120"))
SE_DEMO = int(os.getenv("NWW_QA_SENTS_MIN", "2"))
CH_BASE, SE_BASE = 800, 8

n=demo_ok=base_ok=0
with p.open(encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        ch=int(r.get("num_chars",0)); se=int(r.get("num_sents",0))
        n += 1
        if ch>=CH_DEMO and se>=SE_DEMO: demo_ok += 1
        if ch>=CH_BASE  and se>=SE_BASE: base_ok += 1

def pct(a,b): 
    return f"{(a/max(1,b)):.0%}"

print(f"[overview] n={n} | demo_qa_rate={pct(demo_ok,n)} (chars>={CH_DEMO}, sents>={SE_DEMO}) | baseline_qa_rate={pct(base_ok,n)} (chars>=800, sents>=8)")
