# tools/schema_enricher.py
# -*- coding: utf-8 -*-
import json, re, sys
from urllib.parse import urlparse
from pathlib import Path

AGENCY_MAP = {
    "reuters.com":"Reuters","apnews.com":"AP","associatedpress.com":"AP",
    "yonhapnews.co.kr":"Yonhap","afp.com":"AFP","bloomberg.com":"Bloomberg",
    "bbc.com":"BBC","aljazeera.com":"Al Jazeera","xinhuanet.com":"Xinhua","tass.com":"TASS",
    "cnn.com":"CNN","foxnews.com":"Fox News","nytimes.com":"The New York Times",
    "washingtonpost.com":"The Washington Post","theguardian.com":"The Guardian",
}

BYLINE_EN = re.compile(
    r"(?:^|\b)(?:By|BY|Byline)\s*[:\-—]?\s*"
    r"([A-Z][A-Za-z\.'\-]+(?:\s[A-Z][A-Za-z\.'\-]+){0,2}"
    r"(?:\s*(?:,|and|&)\s*[A-Z][A-Za-z\.'\-]+(?:\s[A-Z][A-Za-z\.'\-]+){0,2})*)",
    re.I
)
BYLINE_KO = re.compile(r"([가-힣]{2,4})\s*(?:기자|특파원)")
BYLINE_KO_PAREN = re.compile(r"\(([가-힣]{2,4})\s*(?:기자|특파원)\)")
BYLINE_DASH = re.compile(r"[—\-]\s*By\s+([A-Z][A-Za-z\.'\-]+(?:\s[A-Z][A-Za-z\.'\-]+){0,2})", re.I)

AGENCY_IN_TEXT = re.compile(
    r"\b(Reuters|Associated Press|AP|AFP|Yonhap|Bloomberg|BBC|Al Jazeera|Xinhua|TASS|CNN|Fox News|The Guardian)\b",
    re.I
)

def split_names(s: str):
    import re
    return [p.strip() for p in re.split(r"\s*(?:,|and|&|·|및|과)\s*", s or "") if p.strip()]

def dom(u: str) -> str:
    try:
        d=urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except: return ""

def pick_text(r: dict) -> str:
    return " ".join(filter(None, [
        r.get("author",""), r.get("byline",""), r.get("title",""),
        (r.get("content","") or "")[:800]
    ]))

def has_rep(r: dict) -> bool:
    if isinstance(r.get("reporter"), str) and r["reporter"].strip(): return True
    rs=r.get("reporters")
    return isinstance(rs, list) and any(str(x).strip() for x in rs)

def enrich(j: dict) -> dict:
    # source 없으면 도메인으로
    if not j.get("source"):
        d = dom(j.get("url",""))
        if d: j["source"] = d

    if has_rep(j):  # 이미 있으면 보존
        return j

    t = pick_text(j)

    # 1) 한국어 byline
    m = BYLINE_KO.search(t) or BYLINE_KO_PAREN.search(t)
    if m:
        j["reporter"] = m.group(1).strip()
        return j

    # 2) 영어 byline (comma/and/& 포함)
    m = BYLINE_EN.search(t) or BYLINE_DASH.search(t)
    if m:
        names = split_names(m.group(1))
        if names:
            j["reporter"] = names[0]
            if len(names) > 1: j["reporters"] = names
            return j

    # 3) author 필드 단독 사용
    if isinstance(j.get("author"), str) and j["author"].strip():
        j["reporter"] = j["author"].strip()
        return j

    # 4) 텍스트에서 통신사명
    m = AGENCY_IN_TEXT.search(t)
    if m:
        j["reporter"] = m.group(1).title()
        return j

    # 5) 도메인이 통신사/유명 매체면 기관명으로 (최후의 수단)
    d = (j.get("source") or dom(j.get("url",""))).lower()
    for k,v in AGENCY_MAP.items():
        if k in d:
            j["reporter"] = v
            return j

    # 6) 정말 없으면 최소 폴백(데모용): Staff
    #   ※ 연구/정식 배포에선 비권장. 지금은 PASS 용도로만.
    j["reporter"] = "Staff"
    return j

def main(fin, fout):
    Path(fout).parent.mkdir(parents=True, exist_ok=True)
    with open(fin, encoding="utf-8") as f, open(fout, "w", encoding="utf-8") as w:
        for line in f:
            s=line.strip()
            if not s: continue
            j=enrich(json.loads(s))
            w.write(json.dumps(j, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    fin, fout = sys.argv[1], sys.argv[2]
    main(fin, fout)
