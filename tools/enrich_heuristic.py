# -*- coding: utf-8 -*-
# data\b01\clean.jsonl -> data\b01\clean.enriched.jsonl
import re, json, tldextract
from pathlib import Path

ROOT=Path("data"); B="b01"
fin = ROOT/B/"clean.jsonl"
fout= ROOT/B/"clean.enriched.jsonl"

# 1) 기본 도메인→언론사 맵
DOMAIN2SRC = {
  # 포털
  "news.naver.com":"네이버","n.news.naver.com":"네이버","m.news.naver.com":"네이버",
  "news.daum.net":"다음","v.daum.net":"다음","news.kakao.com":"카카오",
  "naver.com":"네이버","daum.net":"다음","kakao.com":"카카오",

  # 통신/방송
  "yna.co.kr":"연합뉴스","news1.kr":"뉴스1","newsis.com":"뉴시스",
  "ytn.co.kr":"YTN","mbn.co.kr":"MBN","kbs.co.kr":"KBS","sbs.co.kr":"SBS",
  "mbc.co.kr":"MBC","jtbc.co.kr":"JTBC","tvchosun.com":"TV조선","channela.co.kr":"채널A",

  # 종합/경제
  "chosun.com":"조선일보","donga.com":"동아일보","joongang.co.kr":"중앙일보","joins.com":"중앙일보",
  "hani.co.kr":"한겨레","khan.co.kr":"경향신문","hankookilbo.com":"한국일보","seoul.co.kr":"서울신문",
  "munhwa.com":"문화일보","segye.com":"세계일보","kmib.co.kr":"국민일보",
  "mk.co.kr":"매일경제","hankyung.com":"한국경제","edaily.co.kr":"이데일리",
  "asiae.co.kr":"아시아경제","fnnews.com":"파이낸셜뉴스","heraldcorp.com":"코리아헤럴드",
  "biz.chosun.com":"조선비즈","moneys.co.kr":"머니S",

  # IT/산업/전문/지역 일부
  "etnews.com":"전자신문","zdnet.co.kr":"지디넷코리아","inews24.com":"아이뉴스24","bloter.net":"블로터",
  "ddaily.co.kr":"디지털데일리","thebell.co.kr":"더벨","mediatoday.co.kr":"미디어오늘",
  "pressian.com":"프레시안","ohmynews.com":"오마이뉴스","nocutnews.co.kr":"노컷뉴스",
  "kookje.co.kr":"국제신문","busan.com":"부산일보","kyeongin.com":"경인일보","jejusori.net":"제주의소리",

  # 외신(자주 인용됨)
  "reuters.com":"로이터","apnews.com":"AP통신","afp.com":"AFP","bbc.co.uk":"BBC",
  "cnn.com":"CNN","nytimes.com":"뉴욕타임스","washingtonpost.com":"워싱턴포스트",
  "bloomberg.com":"블룸버그","ft.com":"파이낸셜타임스","theguardian.com":"가디언",
  "aljazeera.com":"알자지라","dw.com":"도이체벨레","nikkei.com":"니케이","japantimes.co.jp":"재팬타임즈",
  "scmp.com":"사우스차이나모닝포스트",

  # 샘플 도메인(테스트용)
  "ex1.":"샘플","ex2.":"샘플"
}

# 2) 선택적 오버라이드 로딩 (rules\domain_override.json)
try:
    _ov = Path("rules/domain_override.json")
    if _ov.exists():
        DOMAIN2SRC.update(json.loads(_ov.read_text(encoding="utf-8")))
except Exception as e:
    print("[enrich] override load failed:", e)

MEDIA_WORDS = list(set(DOMAIN2SRC.values()) | set(["연합뉴스","뉴스1","뉴시스","YTN","KBS","SBS","MBC","JTBC"]))

# 3) 패턴들
REP_PAT  = re.compile(r"([가-힣]{2,4})\s*(?:기자|특파원|인턴기자|선임기자|논설위원|평론가)\b")
REP_EQ   = re.compile(r"([가-힣]{2,4})\s*기자\s*=\s*")
REP_BR   = re.compile(r"\[?([가-힣]{2,4})\s*기자\]?")
NAV_EQ   = re.compile(r"([가-힣]{2,4})\s*기자\s*=\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
NAV_MISC = re.compile(r"\b([가-힣]{2,4})\s*기자\b")
EN_REP   = re.compile(r"\bBy\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:Correspondent|Reporter)\b")
EMAIL    = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)")
DATE     = re.compile(r"(20\d{2})[./\-년]\s*(\d{1,2})[./\-월]\s*(\d{1,2})[일]?(?:\s*[T\s]?\s*(\d{1,2}):(\d{1,2}))?")

def guess_source(url, text):
    url=url or ""; text=text or ""
    for k,v in DOMAIN2SRC.items():
        if k in url:
            if "naver.com" in k:  # 포털이면 본문에서 원출처 우선
                for w in MEDIA_WORDS:
                    if w in text: return w
            return v
    for w in MEDIA_WORDS:
        if w in text: return w
    return None

def find_reporters(text):
    s=text or ""
    reps=set()
    for rx in (REP_PAT, REP_EQ, REP_BR, NAV_EQ, NAV_MISC):
        for m in rx.finditer(s): reps.add(m.group(1)+" 기자")
    m=EN_REP.search(s)
    if m: reps.add(m.group(1)+" Reporter")
    for m in EMAIL.finditer(s):
        left=s[:m.start()]
        m2 = (REP_PAT.search(left[-20:]) or REP_BR.search(left[-20:]) or
              REP_EQ.search(left[-20:]) or NAV_MISC.search(left[-20:]))
        if m2: reps.add(m2.group(1)+" 기자")
    return sorted(reps)

def find_published(text):
    m=DATE.search(text or "")
    if not m: return None
    y,mo,d,hh,mm=m.groups()
    if hh and mm: return f"{int(y):04d}-{int(mo):02d}-{int(d):02d} {int(hh):02d}:{int(mm):02d}"
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

def main():
    if not fin.exists():
        raise FileNotFoundError(f"not found: {fin}")
    n=src_hit=rep_hit=0
    with fin.open(encoding="utf-8") as f, fout.open("w",encoding="utf-8") as w:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            txt=r.get("clean_text") or r.get("text") or r.get("body") or ""
            url=r.get("url") or ""
            r["source"]      = r.get("source") or guess_source(url, txt)
            reps             = r.get("reporters") or r.get("reporter")
            if not reps: reps = find_reporters(txt)
            if isinstance(reps,str): reps=[reps]
            r["reporters"]   = reps or []
            r["published_at"]= r.get("published_at") or find_published(txt)
            src_hit += 1 if r.get("source") else 0
            rep_hit += 1 if r.get("reporters") else 0
            n+=1; w.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"enriched: {n}  source_cov={src_hit/max(1,n):.3f}  reporter_cov={rep_hit/max(1,n):.3f}  -> {fout}")

if __name__=="__main__":
    main()
