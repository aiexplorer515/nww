# -*- coding: utf-8 -*-
"""
기사 수집(ingest) 모듈: URL -> (본문, 메타데이터) 추출
- 1차 시도: trafilatura
- 폴백: requests + BeautifulSoup
"""
from __future__ import annotations
from urllib.parse import urlparse
from datetime import datetime
import re

def _try_trafilatura(url: str) -> tuple[str|None, dict]:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, {}
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )
        meta = {}
        # trafilatura에서 메타 파싱이 버전마다 달라 안전하게 후처리만 유지
        return text, meta
    except Exception:
        return None, {}

def _fallback_bs4(url: str) -> tuple[str|None, dict]:
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                          " AppleWebKit/537.36 (KHTML, like Gecko)"
                          " Chrome/122.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        # title, published 추정
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            ogt = soup.find("meta", property="og:title")
            if ogt and ogt.get("content"):
                title = ogt["content"].strip()

        published = None
        for attr in ["article:published_time", "og:updated_time"]:
            mt = soup.find("meta", property=attr)
            if mt and mt.get("content"):
                published = mt["content"].strip(); break
        if not published:
            for nm in ["datePublished", "pubdate", "publishdate"]:
                mt = soup.find("meta", attrs={"itemprop": nm}) or soup.find("meta", attrs={"name": nm})
                if mt and mt.get("content"):
                    published = mt["content"].strip(); break

        # 본문 추출(스크립트/스타일 제거)
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        return (text or None), {"title": title, "published": published}
    except Exception:
        return None, {}

def get_domain(url: str) -> str:
    return urlparse(url).netloc

def collect(url: str) -> dict:
    """
    URL -> dict(text, title, published, source, url, fetched_at)
    """
    text, meta = _try_trafilatura(url)
    if not text:
        text, meta2 = _fallback_bs4(url)
        meta.update(meta2)

    return {
        "url": url,
        "source": get_domain(url),
        "title": meta.get("title"),
        "published": meta.get("published"),
        "text": text or "",
        "fetched_at": datetime.utcnow().isoformat()
    }

class NewsCollector:
    @staticmethod
    def collect(url: str) -> dict:
        return collect(url)

    @staticmethod
    def get_domain(url: str) -> str:
        return get_domain(url)