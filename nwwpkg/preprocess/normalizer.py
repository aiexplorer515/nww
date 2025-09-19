"""
Normalizer - 뉴스 기사 전처리 모듈
articles.jsonl → articles.norm.jsonl
"""

import os
import re
import json
import hashlib
import logging
from typing import Dict, List
from langdetect import detect, DetectorFactory
import nltk

# 재현성 보장
DetectorFactory.seed = 0

# NLTK tokenizer 다운로드 (최초 1회 필요)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(self):
        self.hash_set = set()  # 중복 검사용 해시 저장소

    def run(self, input_path: str, output_path: str, log_path: str = None):
        """입력 파일 → 정규화 → 출력 파일"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                try:
                    article = json.loads(line.strip())
                    norm_article = self._normalize_article(article)
                    if norm_article:
                        fout.write(json.dumps(norm_article, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"[Normalizer] Failed line: {e}")

        logger.info(f"[Normalizer] Completed. → {output_path}")

        # 로그 기록
        if log_path:
            with open(log_path, "w", encoding="utf-8") as flog:
                flog.write(f"Normalization finished for {input_path}\n")

    def _normalize_article(self, article: Dict) -> Dict:
        """단일 기사 정규화"""
        text = article.get("text", "")
        if not text.strip():
            return None

        # 1) 중복 제거 (hash)
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in self.hash_set:
            return None
        self.hash_set.add(text_hash)

        # 2) 소문자 변환
        norm_text = text.lower()

        # 3) 특수문자 정리
        norm_text = re.sub(r"\s+", " ", norm_text)  # 다중 공백 → 단일 공백
        norm_text = re.sub(r"[^0-9a-zA-Z가-힣 .,!?]", "", norm_text)  # 불필요한 기호 제거

        # 4) 문장 분리
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(norm_text, language="english")
        except:
            sentences = norm_text.split(".")

        # 5) 언어 감지 (원본 text 기준)
        try:
            lang = detect(text)
        except:
            lang = article.get("lang", "unknown")

        # 6) 출력 구조
        norm_article = {
            "id": article.get("id"),
            "ts": article.get("ts"),
            "title": article.get("title", "").strip(),
            "text": " ".join(sentences).strip(),
            "domain": article.get("domain", "unknown"),
            "region": article.get("region", "global"),
            "source": article.get("source"),
            "lang": lang
        }
        return norm_article
