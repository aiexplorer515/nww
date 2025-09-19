# tools/llm_enrich.py
# -*- coding: utf-8 -*-
import os, sys, re, json, time, hashlib
from urllib.parse import urlparse

try:
    import orjson as oj
    def jloads(s): return oj.loads(s)
    def jdumps(o): return oj.dumps(o).decode()
except Exception:
    def jloads(s): return json.loads(s)
    def jdumps(o): return json.dumps(o, ensure_ascii=False)

import requests

PROVIDER = os.getenv("NWW_LLM_PROVIDER","openai").lower()
MODEL    = os.getenv("NWW_LLM_MODEL","gpt-4o-mini")

SYSTEM = """너는 뉴스 메타데이터 추출기다. 한국어 기사 제목/본문에서
- source(언론사, 예: 연합뉴스, 뉴스1, JTBC, 조선일보 등)
- reporters(기자명 리스트, 예: ["홍길동","김기자"])
- published_at(게시시각, ISO8601. 예: 2025-09-17T12:34:00+09:00 또는 2025-09-17T03:34:00Z)
을 JSON ONLY로 추출하라. 모르면 null 또는 빈 리스트를 넣어라.
출력 형식:
{"source": "...", "reporters": ["...","..."], "published_at": "...", "confidence": 0~1, "evidence": ["근거1","근거2"]}"""

USR_TMPL = """기사:
제목: {title}
본문(발췌): {text}

요구사항:
1) source는 언론사명(브랜드명)으로 간단히.
2) reporters는 "홍길동 기자"에서 "기자" 제거, 최대 3명.
3) published_at은 본문/메타에 보이는 시간을 ISO8601로. 타임존이 보이면 그대로(+09:00 등). 보이지 않으면 null.
4) confidence는 0~1.
반드시 JSON만 출력."""

# 도메인 → 매체 간단 맵(필요시 추가)
DOMAIN2SRC = {
    "yna.co.kr":"연합뉴스","yonhapnews.co.kr":"연합뉴스",
    "news1.kr":"뉴스1","ytn.co.kr":"YTN","kbs.co.kr":"KBS","sbs.co.kr":"SBS","mbc.co.kr":"MBC","jtbc.co.kr":"JTBC",
    "hankyung.com":"한국경제","mk.co.kr":"매일경제","chosun.com":"조선일보","joongang.co.kr":"중앙일보","donga.com":"동아일보",
    "hani.co.kr":"한겨레","khan.co.kr":"경향신문","nocutnews.co.kr":"노컷뉴스","newsis.com":"뉴시스","seoul.co.kr":"서울신문"
}

REP_PAT = re.compile(r"([가-힣]{2,4})\s*기자")
ISO_DT_PAT = re.compile(r"(20\d{2}[-./]\d{1,2}[-./]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+\-]\d{2}:\d{2})?)")

def read_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                try: yield jloads(s)
                except: pass

def write_jsonl(p, rows):
    with open(p,"w",encoding="utf-8") as w:
        for r in rows: w.write(jdumps(r) + "\n")

def call_openai(prompt:str)->str:
    base = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1").rstrip("/")
    url  = f"{base}/chat/completions"
    headers={"Authorization":f"Bearer {os.getenv('OPENAI_API_KEY','')}",
             "Content-Type":"application/json"}
    data={"model":MODEL,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
          "temperature":0.0}
    r=requests.post(url,json=data,headers=headers,timeout=60); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_ollama(prompt:str)->str:
    url="http://localhost:11434/api/generate"
    data={"model":MODEL,"prompt":SYSTEM+"\n\n"+prompt,"stream":False,"options":{"temperature":0.0}}
    r=requests.post(url,json=data,timeout=90); r.raise_for_status()
    return r.json()["response"]

def extract_llm(title:str, text:str):
    prompt = USR_TMPL.format(title=title or "", text=(text or "")[:3500])
    raw = call_ollama(prompt) if PROVIDER=="ollama" else call_openai(prompt)
    # 코드블럭 제거 & json만 추출
    s = raw.strip()
    if "```" in s:
        parts = s.split("```")
        # 마지막 코드블럭 우선
        cand = None
        for chunk in reversed(parts):
            chunk = chunk.strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                cand = chunk; break
        s = cand or s
    try:
        out = jloads(s)
        if not isinstance(out, dict): raise ValueError("not dict")
    except Exception:
        out = {"source":None,"reporters":[],"published_at":None,"confidence":0.0,"evidence":[]}
    # 정규화
    src = (out.get("source") or None)
    reps= out.get("reporters") or []
    if isinstance(reps, str): reps=[reps]
    reps=[re.sub(r"\s*기자$","", x).strip() for x in reps if isinstance(x,str) and x.strip()]
    pub = out.get("published_at") or None
    # published_at 간단 보정: YYYY-MM-DD 또는 시간만 있으면 그대로 두고, Z/offset 없으면 None 유지
    conf= float(out.get("confidence") or 0.0)
    ev  = out.get("evidence") or []
    if isinstance(ev,str): ev=[ev]
    return {"source":src,"reporters":reps,"published_at":pub,"confidence":conf,"evidence":ev}

def fallback_heur(title:str, text:str, url:str, orig_date:str):
    # 1) source: 도메인 맵
    src=None
    try:
        host = urlparse(url or "").netloc.lower()
        src = DOMAIN2SRC.get(host) or DOMAIN2SRC.get(".".join(host.split(".")[-2:]))
    except: pass
    # 2) reporters: "홍길동 기자"
    reps=set()
    for m in REP_PAT.finditer((title or "")+" "+(text or "")):
        reps.add(m.group(1))
        if len(reps)>=3: break
    # 3) published_at: ISO 패턴 or 원본 date(있으면 사용)
    pub=None
    m = ISO_DT_PAT.search((text or "")+" "+(title or ""))
    if m: pub = m.group(1)
    elif orig_date and isinstance(orig_date,str):
        # orig_date가 이미 ISO 비슷하면 그대로 사용
        pub = orig_date if re.match(r"^20\d{2}-\d{2}-\d{2}T", orig_date) else None
    conf = 0.5 if (src or reps or pub) else 0.0
    return {"source":src,"reporters":list(reps),"published_at":pub,"confidence":conf,"evidence":[]}

def merge_pref(llm, fb):
    # LLM 우선, 비면 fallback 보강
    src = llm["source"] or fb["source"]
    reps= llm["reporters"] or fb["reporters"]
    pub = llm["published_at"] or fb["published_at"]
    conf= max(float(llm.get("confidence",0.0)), float(fb.get("confidence",0.0)))
    ev  = (llm.get("evidence") or []) + (fb.get("evidence") or [])
    return {"source":src,"reporters":reps,"published_at":pub,"confidence":round(conf,3),"evidence":ev}

def main(fin, fout, merge_out=None, limit=None):
    rows=[]
    for i, r in enumerate(read_jsonl(fin)):
        if limit and i>=limit: break
        title = r.get("title") or ""
        text  = r.get("clean_text") or r.get("text") or r.get("body") or ""
        url   = r.get("url") or ""
        orig_date = r.get("date") or r.get("t_on") or r.get("time")
        t1=time.time()
        try:
            llm = extract_llm(title, text)
        except Exception:
            llm = {"source":None,"reporters":[],"published_at":None,"confidence":0.0,"evidence":[]}
        fb  = fallback_heur(title, text, url, orig_date)
        meta= merge_pref(llm, fb)
        meta.update({
            "id": r.get("id"),
            "url": url,
            "provider": PROVIDER, "model": MODEL,
            "latency_ms": int((time.time()-t1)*1000)
        })
        rows.append(meta)

    write_jsonl(fout, rows)
    print(f"enrich -> {fout} n={len(rows)}")

    # 병합본 생성(옵션)
    if merge_out:
        base = {r.get("id"):r for r in read_jsonl(fin)}
        for e in rows:
            b = base.get(e["id"])
            if not b: 
                base[e["id"]] = {"id": e["id"], "url": e["url"]}
                b = base[e["id"]]
            # 빈 값이면 대체, 값 있으면 보존
            if not b.get("source") and e.get("source"): b["source"] = e["source"]
            if not b.get("reporter") and e.get("reporters"): b["reporter"] = ", ".join(e["reporters"])
            if not b.get("published_at") and e.get("published_at"): b["published_at"] = e["published_at"]
        # 안정적 출력
        merged = [base[k] for k in base]
        write_jsonl(merge_out, merged)
        print(f"merged -> {merge_out} n={len(merged)}")

if __name__=="__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/llm_enrich.py <in_clean.jsonl> <out_enrich.jsonl> [--merge <out_clean.enriched.jsonl>] [--limit N]")
        sys.exit(1)
    fin = sys.argv[1]; fout = sys.argv[2]
    merge_out=None; limit=None
    if "--merge" in sys.argv:
        merge_out = sys.argv[sys.argv.index("--merge")+1]
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit")+1])
    main(fin, fout, merge_out, limit)
