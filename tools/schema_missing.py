# -*- coding:utf-8 -*-
import json, collections, tldextract
from pathlib import Path
p=Path("data/b01/clean.jsonl")
miss_src, miss_rep, dom_cnt = [], [], collections.Counter()
for line in p.open(encoding="utf-8-sig"):
    r=json.loads(line); url=r.get("url",""); txt=r.get("clean_text","")
    try:
        ext=tldextract.extract(url); dom=".".join([ext.domain,ext.suffix])
    except: dom=""
    dom_cnt[dom]+=1
    if not r.get("source"):      miss_src.append((dom,url))
    if not r.get("reporters"):   miss_rep.append((dom,url))
print("[source 미채움 top 도메인]", collections.Counter(d for d,_ in miss_src).most_common(10))
print("[reporter 미채움 top 도메인]", collections.Counter(d for d,_ in miss_rep).most_common(10))
print("[전체 top 도메인]", dom_cnt.most_common(10))
