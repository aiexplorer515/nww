"""
Extractor - 뉴스 기사 URL / 파일 기반 본문 추출기
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict

class Extractor:
    def __init__(self, bundle_dir: str):
        self.bundle_dir = bundle_dir
        os.makedirs(bundle_dir, exist_ok=True)

    def run(self, output_path: str):
        """sources.csv 또는 urls.txt에서 URL 읽고 본문 추출"""
        urls = self._load_sources()
        articles = []

        for i, url in enumerate(urls):
            try:
                article = self._extract_article(url, i)
                if article:
                    articles.append(article)
            except Exception as e:
                print(f"[ERROR] Failed to extract {url}: {e}")

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            for art in articles:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")

        print(f"[INFO] Extracted {len(articles)} articles → {output_path}")

    def _load_sources(self) -> List[str]:
        """sources.csv 또는 urls.txt에서 URL 목록 불러오기"""
        urls = []
        csv_path = os.path.join(self.bundle_dir, "sources.csv")
        txt_path = os.path.join(self.bundle_dir, "urls.txt")

        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            if "URL" in df.columns:
                urls.extend(df["URL"].dropna().tolist())

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                urls.extend([line.strip() for line in f if line.strip()])

        return list(set(urls))  # 중복 제거

    def _extract_article(self, url: str, idx: int) -> Dict:
        """단일 뉴스 기사 URL에서 본문과 메타데이터 추출"""
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # 제목
        title = soup.title.string if soup.title else f"Article {idx}"

        # 본문 (p 태그 기준 단순 추출)
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        if not text or len(text) < 100:
            raise ValueError("본문 추출 실패")

        # 메타데이터
        article = {
            "id": f"a{idx+1}",
            "ts": datetime.utcnow().isoformat() + "Z",
            "title": title,
            "text": text,
            "domain": "unknown",
            "region": "global",
            "source": url,
            "lang": "ko" if self._is_korean(text) else "en"
        }
        return article

    def _is_korean(self, text: str) -> bool:
        """간단한 언어 감지 (한글 비율 > 20% → 한국어)"""
        korean_chars = sum(1 for ch in text if '가' <= ch <= '힣')
        return korean_chars / max(len(text), 1) > 0.2
