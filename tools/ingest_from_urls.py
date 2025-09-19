# -*- coding: utf-8 -*-
import sys, json, hashlib, tldextract, datetime
from pathlib import Path

# 사용법: python tools\ingest_from_urls.py urls.txt data\b01\ingest.jsonl
urls_txt = Path(sys.argv[1])
out_path = Path(sys.argv[2])

# 기존 URL 중복 방지
seen=set()
if out_path.exists():
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line).get("url",""))
            except: pass

# 우선 trafilatura, 폴백 newspaper3k
def fetch_one(url:str):
    title=author=None; text=None; published=None
    try:
        import trafilatura as tf
        raw = tf.fetch_url(url)
        if raw:
            meta = tf.extract(raw, include_links=False, include_comments=False, with_metadata=True, target_language=None)
            if meta:
                # meta는 str일 수 있음 → json 찾아보기
                import json as _json
                try:
                    data=_json.loads(meta) if isinstance(meta,str) and meta.startswith("{") else None
                except Exception:
                    data=None
                if data:
                    text=data.get("text")
                    title=data.get("title")
                    author=", ".join(data.get("authors") or []) or None
                    published=data.get("date")
                else:
                    text=meta  # 텍스트만 반환된 경우
    except Exception:
        pass
    if not text:
        try:
            from newspaper import Article
            art=Article(url)
            art.download(); art.parse()
            text=art.text or None
            title=title or (art.title or None)
            if art.authors:
                author=", ".join(art.authors)
            if getattr(art,"publish_date",None):
                published=str(art.publish_date.date())
        except Exception:
            pass
    return title, author, text, published

added=skip=0
out_path.parent.mkdir(parents=True, exist_ok=True)
with urls_txt.open(encoding="utf-8") as f, out_path.open("a", encoding="utf-8") as w:
    for line in f:
        url=line.strip()
        if not url or url.startswith("#"): continue
        if url in seen: 
            skip+=1; continue
        tit, auth, txt, pub = fetch_one(url)
        if not txt or len(txt.strip())<300:  # 너무 짧으면 스킵
            skip+=1; continue
        ext=tldextract.extract(url); domain=".".join([ext.domain, ext.suffix]) if ext.suffix else ext.domain
        rid=hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        rec={
            "id": rid,
            "url": url,
            "source": domain,   # 후속 enrich에서 정교화
            "title": tit,
            "text": txt,
            "reporters": [auth] if auth else [],
            "published": pub or None,
            "ingested_at": datetime.datetime.utcnow().isoformat()+"Z"
        }
        w.write(json.dumps(rec, ensure_ascii=False)+"\n")
        added+=1

print(f"ingest appended: +{added}, skipped:{skip}, out={out_path}")
